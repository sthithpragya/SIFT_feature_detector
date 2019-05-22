#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <limits>
 
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d; // import the SURF library that is provided in the opencv_contrib as free

/* We decalre global variables that can be used everywhere */
Rect roi = Rect(Point(24,44),Point(196,210));        // variable that defines the rectangle for the object of interest
std::vector<KeyPoint> keypoints_object;              // will contain the keypoint for object of  interest 
Mat descriptors_object;                              // will contain the descriptors for object of  interest 
Mat objectFrame;                                     // will contain the roi extracted from the image
std::vector<Point2f> object_corners(4);              // will contain the corner position of the objectFrame later used for computing transforms

const int minHessian = 400;                          // Threshold used to classify if a interest point is should be used as a keypoint                 
int frameNumber = 0;                                 // Keep track of the frame number being processed


/* - Make a roi rectangle variable for current frame, given the scene_corners 
   - scene_corners has 4 corner of the polygon that surrounds the region of interest 
   - use it to define a bounding box with help of centroid, width and height extracted 
     from the scene_corners
*/
void makeROI(std::vector<Point2f>& scene_corners,Rect& roi_current){
	
	// Thresholding the values of the corners for valid values
    for(int i =0; i < 4; i++){
      if(scene_corners[i].x <= 0){
        scene_corners[i].x = 1;
      }
      if(scene_corners[i].x >= 320){
        scene_corners[i].x = 319;
      }
      if(scene_corners[i].y <= 0){
        scene_corners[i].y = 1;
      }
      if(scene_corners[i].y >= 240){
        scene_corners[i].y = 239;
      }
    }

    // Calculate min and max for both x and y direction
    Point2f min_first(DBL_MAX,DBL_MAX);
    Point2f max_second(DBL_MIN,DBL_MIN);

    for(int i =0; i < 4; i++){
	    if(min_first.x > scene_corners[i].x){

	      min_first.x = scene_corners[i].x;
	    }
	    if(min_first.y > scene_corners[i].y){
	      min_first.y = scene_corners[i].y;
	    }

	    if(max_second.x < scene_corners[i].x){

	      max_second.x = scene_corners[i].x;
	    }
	    if(max_second.y < scene_corners[i].y){
	      max_second.y = scene_corners[i].y;
	    }
    }

    // Calculate centroid, height and width of the bounding box
    Point2f centroid;
    double width, height;

    centroid.x = (min_first.x + max_second.x)/2;
    centroid.y = (min_first.y + max_second.y)/2;

    width = norm(scene_corners[0]- scene_corners[1]);
    height = norm(scene_corners[1] - scene_corners[2]);

    Point2f p1,p2; 

    p1.x = centroid.x - width/2; 
    p1.y = centroid.y - height/2;

    p2.x = centroid.x + width/2; 
    p2.y = centroid.y + height/2;

    if(p1.x < 0) p1.x = 0;
    if(p1.y < 0) p1.y = 0;

    if(p2.x > 320) p1.x = 320;
    if(p2.y > 240) p1.y = 240;

    // assign the rectangle object to roi_current
    roi_current = Rect(p1,p2);
}

void processObjectFrame(Mat& firstFrame){
    objectFrame = firstFrame(roi);

    Ptr<SURF> detectorFirstFrame = SURF::create();
    detectorFirstFrame->setHessianThreshold(minHessian);
    
    detectorFirstFrame->detect(objectFrame, keypoints_object);
    detectorFirstFrame->compute(objectFrame, keypoints_object, descriptors_object);

    imshow( "Extracted Object", objectFrame);

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
    if( matches[i].distance <= 0.25) //max(2*min_dist, 0.02) )
      { 
        good_matches.push_back( matches[i]); 
      }
  }
}

/* - Handles all the subsequent frame in the video after the first frame 
   - All the subsequent frames have the same procedure
      - generate keypoints and descriptor for thw whole image 
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
    // ignore them they frame didn't have any good keypoint matching
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


	// Show the images with keypoint matches and the trasformation lines
    imshow( "Good Matches", img_matches );

    // Generate the roi from the scene_corners calaculated above 
    Point2f centroid(0,0);
    double width, height;
    Rect roi_current;
    makeROI(scene_corners,roi_current);

    // Exract the object of interest in the current frame
    Mat extraxted_object;
    try
    {
      extraxted_object = currentFrame(roi_current);
      imshow( "Extracted Object", extraxted_object);
    }
    catch ( const std::exception& r_e )
    {
    return;
    }

    // Update the global variable with the roi of the current frame
    roi = roi_current;
    // Update the objectFrame Information
    processObjectFrame(currentFrame);

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
      processObjectFrame(currentframe);
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
