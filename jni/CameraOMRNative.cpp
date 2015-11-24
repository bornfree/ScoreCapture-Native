#include <jni.h>
#include <android/log.h>
#include <string.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define LOG_TAG "CameraOMR NDK"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

//#define SECTION_NUM_ANSWERS 10
#define SECTION_OPTIONS 5

extern "C"{

	struct TemplateX // Dont interfere with template keyword
	{
		int width;
		int height;
	}main_template;

	struct Section
	{
		int left;
		int top;
		int width;
		int height;
		int num_answers;
		char* answers;
	};

	vector<Section> main_sections;


	/*
		Find the corner closest to the origin specified.
		Example: To find Left Bottom marker, we calculate
		distances of blobs in left bottom quadrant of the sheet
		and find the one closest to left bottom boundary.
	*/
	Point2f bestCorner(Mat img, Point2f origin, Ptr<SimpleBlobDetector>& detector)
	{
		std::vector<KeyPoint> keypoints;
		detector->detect(img,keypoints);
		Point2f best_corner;

		double least_distance = img.cols*img.cols + img.rows*img.rows;
		Mat img_clone = img.clone();
		for(int i=0; i < keypoints.size(); i++)
		{
			KeyPoint k = keypoints[i];

			double distance = norm(k.pt-origin);
			if(distance < least_distance)
			{
				best_corner = k.pt;
				least_distance = distance;
			}
		}
		return best_corner;
	}

	/*
		Detect the corner markers and return them in the order tl, tr, br, bl
	*/
	void getCorners(Mat bw, vector<Point2f>& corners)
	{

		SimpleBlobDetector::Params params;
		params.filterByArea = true;
		params.minArea = 200;
		params.filterByCircularity = true;
		params.minCircularity = 0.4;
		params.filterByInertia = true;
		params.minInertiaRatio = 0.5;
		params.filterByConvexity = true;
		params.minConvexity = 0.5;
		// ... any other params you don't want default value

		// set up and create the detector using the parameters
		Ptr<SimpleBlobDetector> corner_detector = SimpleBlobDetector::create(params);

		int x_step = (int)(bw.cols/2.0);
		int y_step = (int)(bw.rows/2.0);

		Point2f corner;

		//  No adjustment needed
		Mat tl = bw(Rect(0,0,x_step, y_step));
		corner = bestCorner(tl, Point2f(0,0), corner_detector);
		corners.push_back(corner);


		// Adjusted according to the quadrant
		Mat tr = bw(Rect(bw.cols-x_step,0,x_step, y_step));
		corner = bestCorner(tr, Point2f(tr.cols,0), corner_detector);
		corner.x += (bw.cols-x_step);
		corners.push_back(corner);

		// Adjusted according to the quadrant
		Mat br = bw(Rect(bw.cols-x_step, bw.rows-y_step, x_step, y_step));
		corner = bestCorner(br, Point2f(br.cols,br.rows), corner_detector);
		corner.x += (bw.cols-x_step);
		corner.y += (bw.rows-y_step);
		corners.push_back(corner);

		// Adjusted according to the quadrant
		Mat bl = bw(Rect(0, bw.rows-y_step, x_step, y_step));
		corner = bestCorner(bl, Point2f(0,bl.rows), corner_detector);
		corner.y+= (bw.rows-y_step);
		corners.push_back(corner);

	}



	Mat perspective_correct(Mat bw)
	{
		//Mat bw;
		//cvtColor(img,bw, CV_BGR2GRAY);

		// Get the corners in the order of tl, tr, br, bl

		vector<Point2f> corners;
		getCorners(bw, corners);


		Mat img_clone = bw.clone();
		for(int i=0; i< corners.size(); i++)
		{
			circle(img_clone, corners[i], 20, Scalar(255,0,0), 5);

		}
		//debugImshow("Alignment markers", img_clone);


		// Define the destination image
		Mat quad = Mat::zeros(main_template.height, main_template.width, CV_8UC1);

		// Corners of the destination image
		vector<Point2f> quad_pts;
		quad_pts.push_back(Point2f(0, 0));
		quad_pts.push_back(Point2f(quad.cols, 0));
		quad_pts.push_back(Point2f(quad.cols, quad.rows));
		quad_pts.push_back(Point2f(0, quad.rows));

		// Get transformation matrix
		Mat transmtx = getPerspectiveTransform(corners, quad_pts);

		// Apply perspective transformation
		warpPerspective(bw, quad, transmtx, quad.size());

		return quad;
	}

	bool cmp(KeyPoint a,KeyPoint b ) {
	    return a.pt.y < b.pt.y ;
	}

	bool cmp_area(KeyPoint a,KeyPoint b ) {
	    return a.size < b.size ;
	}

	Mat markOMR(Mat img, int num_answers, char* answer_key, int &total_score)
	{
		LOGD("Key: %s Length: %d", answer_key, strlen(answer_key));
		cv::SimpleBlobDetector::Params params;
		//params.filterByArea = true;
		//params.minArea = 200;
		params.filterByCircularity = true;
		params.minCircularity = 0.4;
		params.filterByInertia = true;
		params.minInertiaRatio = 0.7;
		params.filterByConvexity = true;
		params.minConvexity = 0.5;
		// ... any other params you don't want default value

		// set up and create the detector using the parameters
		Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
		// Set up the detector with default parameters.
		// SimpleBlobDetector detector;

		// Detect blobs.
		std::vector<KeyPoint> keypoints;
		detector->detect(img,keypoints);

		if(keypoints.size() < 1)
		{
			LOGD("No keypoints found");
			return img;
		}

		// Sort blobs by y coord
		sort( keypoints.begin(), keypoints.end(), cmp );

		// Init answers vector as vector of keypoints
		std::vector< std::vector<KeyPoint> > answers;
		answers.push_back( vector<KeyPoint>() ); //Init with empty

		std::vector<char> final_answers;

		int i=0, row = 0, kp_index=0 ;
		int answer_height = (int) img.rows/(1.0 * num_answers);
		int answer_width = (int) img.cols/(1.0 * SECTION_OPTIONS);

		KeyPoint k = keypoints.at(kp_index);
		bool keypoints_available = true;

		std::vector<KeyPoint> selected_answers;

		Mat img_clone = img.clone();
		int score = 0;
		LOGD("\n--------------------\n");
		// Loop over height of section incrementing each time by answer_height
		while(i < img.rows)
		{
			if(keypoints_available && k.pt.y > i && k.pt.y < (i+answer_height))
			{
				// Falls in this row.
				// Add it and move to next keypoint
				answers[row].push_back(k);
				kp_index++;
				// cout << "Adding KP to row " << row << endl;
			}else
			{
				// Get answer of row
				if(answers[row].size() > 0)
				{
					sort( answers[row].begin(), answers[row].end(), cmp_area );
					char best_answer = 65 + answers[row].front().pt.x/answer_width;
					final_answers.push_back(best_answer);
					selected_answers.push_back(answers[row].front());
					if(best_answer == answer_key[row])
						score++;
				}else
				{
					final_answers.push_back('_');
				}

				// Debug print
				// if(answers[row].size() > 0)
				// {
				// 	cout << "Try printing row " << row << " with elements " << answers[row].size() << endl;
				// 	Mat imw;
				// 	drawKeypoints( img_clone, answers[row], imw, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
				// 	imshow("keypoints", imw );
				// 	waitKey(0);
				// }

				// Not in this row
				// Increment row and initialize with empty vector
				row++;
				answers.push_back( vector<KeyPoint>() );
				i += answer_height;
				// cout << "Moving to next row " << row << endl;
			}

			// Ensure keypoints are still available
			if(kp_index < keypoints.size())
			{
				k = keypoints.at(kp_index);
			}else
			{
				// If this is hit, all subsequent answer vectors will by empty vectors
				// But it WILL count to SECTION_NUM_ANSWERS
				keypoints_available = false;
			}
			// Last section
			if((img.rows - i) < answer_height)
				break;
		}

		for(int i=0; i< final_answers.size(); i++)
		{
			if(final_answers[i] == '_')
				LOGD("%d : \n", i);//cout << i << " : " <<endl;
			else
				LOGD("%d : %c \n",i, final_answers[i]);//cout << i << " : " << string(2*(final_answers[i]-65), ' ') << final_answers[i] <<endl;
		}
		LOGD("\n--------------------\n");//cout << endl << "--------------------" << endl;
		LOGD("Section Score : %d\n--------------------\n", score);//cout << "Section score : "<< score << endl << "--------------------" << endl;
		total_score += score;
		// Draw detected blobs as red circles.
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
		Mat img_with_keypoints;
		drawKeypoints( img, selected_answers, img_with_keypoints, Scalar(0,0,0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS );
		return img_with_keypoints;
	}

	bool hasPattern(Mat mat)
	{
	  bool hasPatternValue = false;
	  int pattern_square_size = 10;
	  int mid_point = (int) mat.cols/2.0;
	  String patternString = "";

	  for(int i = -3*pattern_square_size; i < 3*pattern_square_size; i+=pattern_square_size)
	  {
		Mat square_area = mat(Rect(mid_point+i, mat.rows - pattern_square_size, pattern_square_size, pattern_square_size));
		Mat square = square_area.clone();
		threshold(square, square, 127, 255, CV_THRESH_BINARY);
		int whites = countNonZero(square);
		if(whites > (square.cols*square.rows - whites))
		  patternString += "w";
		else
		  patternString += "b";

	  }

	  String goodPattern = "wbwbwb";
	  //LOGD("Found pattern %s", patternString.c_str());
	  if(goodPattern.compare(patternString) == 0)
	    hasPatternValue = true;

	  return hasPatternValue;
	}

	JNIEXPORT void JNICALL Java_com_cameraomr_android_CameraActivity_setTemplateProperties(JNIEnv* env, jobject, jint template_width, jint template_height)
	{
		main_template.width = template_width;
		main_template.height = template_height;

		LOGD("Template w: %d h: %d", main_template.width, main_template.height);
	}

	JNIEXPORT void JNICALL Java_com_cameraomr_android_CameraActivity_setSections(JNIEnv* env, jobject, jobjectArray sectionsArray)
	{
		 main_sections.clear();
		 int len = env->GetArrayLength(sectionsArray);
		 jclass jSectionNDKClass = env->FindClass("com/cameraomr/android/classes/SectionNDK");

		 struct Section section;

		 for(int i=0; i < len; ++i)
		{
			 jobject sectionObject = (jobject) env->GetObjectArrayElement(sectionsArray, i);
			 section.width  = env->GetIntField(sectionObject, env->GetFieldID(jSectionNDKClass, "width", "I"));
			 section.height = env->GetIntField(sectionObject, env->GetFieldID(jSectionNDKClass, "height", "I"));
			 section.left   = env->GetIntField(sectionObject, env->GetFieldID(jSectionNDKClass, "left", "I"));
			 section.top    = env->GetIntField(sectionObject, env->GetFieldID(jSectionNDKClass,  "top", "I"));
			 section.num_answers = env->GetIntField(sectionObject, env->GetFieldID(jSectionNDKClass, "num_answers", "I"));

			 jstring str = (jstring) env->GetObjectField(sectionObject, env->GetFieldID(jSectionNDKClass, "answers", "Ljava/lang/String;"));
			 const char* answers_string = env->GetStringUTFChars(str, 0);
			 section.answers = strdup(answers_string);

			 env->ReleaseStringUTFChars(str, answers_string);


			 LOGD("w: %d h: %d l: %d t : %d a: %d ", section.width, section.height, section.left, section.top, section.num_answers);
			 LOGD("Answers: %s", section.answers);

			 main_sections.push_back(section);

		}
	}

	JNIEXPORT jint JNICALL Java_com_cameraomr_android_classes_Frame_processFrame(JNIEnv* env, jobject, jlong addrGray, jint debugIndex)
		{

		  int total_score = 0;
		  /* debugIndex
		   * 0 = Normal Grayscale
		   * 1 = Perspective Corrected
		   * 2 = Section 1 marked
		   * 3 = Section 2 marked
		   * */

		  Mat& mGray = *(Mat*)addrGray;

		  // We need to rotate the Mat by 90 degree clockwise
		  // Required as OpenCV sees the camera data rotated
		  transpose(mGray, mGray);
		  flip(mGray, mGray, 1);

		  Mat result = perspective_correct(mGray);
		  //if(debugIndex == 1)
		  mGray = result;
		  if(!hasPattern(result))
		  {
			  LOGD("No pattern found");
			  return -1;
		  }


		  for(int i=0; i < main_sections.size(); i++)
		  {
			  Section section = main_sections[i];
			  Mat sectionMat = result(Rect(section.left, section.top, section.width, section.height));
			  Mat sectionROI =  markOMR(sectionMat, section.num_answers, section.answers, total_score);
			  if(debugIndex > 1)
				  mGray = sectionROI;
		  }

		  jint jtotal_score = total_score;
		  return jtotal_score;
		}


}
