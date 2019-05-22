#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <limits>
 
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d; // SURF library provided in the opencv_contrib as free

/* We decalre global variables that can be used everywhere */
Rect roi = Rect(Point(24,44),Point(196,210));        // variable that defines the rectangle for the object of interest in frame 1
std::vector<KeyPoint> keypoints_object;              // will contain the keypoint for object of  interest 
Mat descriptors_object;                              // will contain the descriptors for object of  interest 
Mat objectFrame;                                     // will contain the roi extracted from the image
std::vector<Point2f> object_corners(4);              // will contain the corner position of the objectFrame later used for computing transforms

const int minHessian = 400;                          // Threshold used to classify if a interest point is should be used as a keypoint                 
int frameNumber = 0;                                 // Keep track of the frame number being processed



/* - Processing the First Frame differently as it constains the object of interest.
   - We extract all the information required to find the matches and transformation 
     for the subsequent frames. 
   - All these values are stored in global variable declared above  
*/
void processFirstFrame(Mat& firstFrame){

    objectFrame = firstFrame(roi);                          // Extract just the image of the object with help of roi defined 

    Ptr<SURF> detectorFirstFrame = SURF::create();          // create object for SURF class 
    detectorFirstFrame->setHessianThreshold(minHessian);    // set the threshold for a interest point to be treated as a keypoint
    
    // detect the keypoints and compute the descripotors for each keypoint 
    detectorFirstFrame->detect(objectFrame, keypoints_object);  
    detectorFirstFrame->compute(objectFrame, keypoints_object, descriptors_object);

    imshow( "Extracted Object", objectFrame);

    // store the corners of the objectFrame that will be later used to compute the homography
    object_corners[0] = cvPoint(0,0); 
    object_corners[1] = cvPoint( objectFrame.cols, 0 );
   	object_corners[2] = cvPoint( objectFrame.cols, objectFrame.rows ); 
   	object_corners[3] = cvPoint( 0, objectFrame.rows );

}


/*  - Given a set of matches, only keep those matches that are less than certain threshold 
    - This will help us remove maching of keypoints who's descriptors are not close to each other
*/
void calculateGoodMatches(std::vector< DMatch >& matches,  std::vector< DMatch >& good_matches){

  // Keep matches that have distance less than 0.25, this value has been defined experimentally 
  // 0.25 has been choosen such that we get very close correspondences of keypoint and at the same 
  // time we always have some matches
  for( int i = 0; i < descriptors_object.rows; i++ )
  { 
    if( matches[i].distance <= 0.25) 
      { 
        good_matches.push_back( matches[i]); 
      }
  }
  
}

/* - Handles all the subsequent frames in the video post the first frame 
   - All the subsequent frames follow the same procedure
      - generate keypoints and descriptor for the whole image 
      - match them to the objects of interest's keypoint and descriptor
        as extracted during the First frame processing (stored in global variable)
      - calculate good matches from the generated matches (strong correspondences)
*/ 
void calculateAndMatch(Mat& currentFrame){

    std::vector<KeyPoint> keypoints_frame;
    Mat descriptors_frame;

    Ptr<SURF> detector = SURF::create();
    detector->setHessianThreshold(minHessian);
    
    detector->detect(currentFrame, keypoints_frame);
    detector->compute(currentFrame,keypoints_frame, descriptors_frame );

    
    // Matching descriptor vectors using FLANN matcher
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( descriptors_object, descriptors_frame, matches );

  std::vector< DMatch > good_matches;
  calculateGoodMatches(matches,good_matches);

  if(good_matches.size() == 0){
    // ignore the frames which don't have any good keypoint matching
  }
  else{
    // Draw only strong matches
    Mat img_matches;
    drawMatches( objectFrame, keypoints_object, currentFrame, keypoints_frame,
                 good_matches, img_matches);

	  // Localize the object
	  std::vector<Point2f> obj;
	  std::vector<Point2f> scene;

	  for( int i = 0; i < good_matches.size(); i++ )
	  {
	    // Get the keypoints from the good matches
	    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
	    scene.push_back( keypoints_frame[ good_matches[i].trainIdx ].pt );
	  }

    // Compute the homograpgy with help of keypoint correspondences between objectFrame and given frame
	  Mat H = findHomography( obj, scene, CV_RANSAC );

    // Compute the perspectiveTransform given H and corners in the object Frame
	  std::vector<Point2f> scene_corners(4);
	  perspectiveTransform( object_corners, scene_corners, H);

	  // Draw lines between the corners (the mapped object in the scene - image_2 )
	  line( img_matches, scene_corners[0] + Point2f( objectFrame.cols, 0), scene_corners[1] + Point2f( objectFrame.cols, 0), Scalar(0, 255, 0), 4 );
	  line( img_matches, scene_corners[1] + Point2f( objectFrame.cols, 0), scene_corners[2] + Point2f( objectFrame.cols, 0), Scalar( 0, 255, 0), 4 );
	  line( img_matches, scene_corners[2] + Point2f( objectFrame.cols, 0), scene_corners[3] + Point2f( objectFrame.cols, 0), Scalar( 0, 255, 0), 4 );
	  line( img_matches, scene_corners[3] + Point2f( objectFrame.cols, 0), scene_corners[0] + Point2f( objectFrame.cols, 0), Scalar( 0, 255, 0), 4 );

	  // Show the umages with keypoint matches and the trasformation lines
    imshow( "Good Matches", img_matches );
  }

}

int main(){
    
  // Create a VideoCapture object and open the input file
  // If the input is the web camera, pass 0 instead of the video file name
  VideoCapture cap("../video1.mp4");
    
  // Check if camera opened successfully
  if(!cap.isOpened()){
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
   
     
  while(1){
 
    Mat currentframe;
    // Capture frame-by-frame
    cap >> currentframe;
  
    // If the frame is empty, break immediately
    if (currentframe.empty())
      break;

    // 1. convert the images from the video in a grey scale 
    cvtColor(currentframe,currentframe,COLOR_BGR2GRAY);
    imshow( "Frame", currentframe );


    if(frameNumber != 0){
      // currentframe is not the first frame
      calculateAndMatch(currentframe);
    }
    else{
      // currentframe is the first frame 
      processFirstFrame(currentframe);
    }
    
    // Press  ESC on keyboard to exit
    char c=(char)waitKey(25);
    if(c==27)
      break;

    frameNumber++;

  }
  
  // When everything done, release the video capture object
  cap.release();
 
  // Closes all the frames
  destroyAllWindows();
     
  return 0;
}