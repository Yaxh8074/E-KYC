import cv2
import time
import argparse
from liveness_detection import LivenessDetection

def main():
    parser = argparse.ArgumentParser(description='Test liveness detection')
    parser.add_argument('--webcam', type=int, default=0, help='Webcam index')
    args = parser.parse_args()
    
    # Initialize liveness detector
    liveness_detector = LivenessDetection()
    
    # Initialize webcam
    cap = cv2.VideoCapture(args.webcam)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting liveness detection demo...")
    print("Press 'q' to quit")
    print("Press 's' to test single frame liveness check")
    print("Press 'c' to test challenge-response liveness check")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Display instructions
        cv2.putText(
            frame, 
            "Press 's' for static check, 'c' for challenge, 'q' to quit", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 0), 
            2
        )
        
        # Display frame
        cv2.imshow("Liveness Detection Demo", frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Perform static liveness check
            is_live, confidence, details = liveness_detector.check_liveness(frame)
            
            # Display results
            print("\n--- Static Liveness Check Results ---")
            print(f"Liveness: {'Passed' if is_live else 'Failed'}")
            print(f"Confidence: {confidence}%")
            print(f"Details: {details}")
            
        elif key == ord('c'):
            print("\nStarting challenge-response test...")
            
            # Define callbacks for challenge-response test
            def capture_callback():
                ret, frame = cap.read()
                if ret:
                    # Display frame
                    cv2.imshow("Liveness Detection Demo", frame)
                    cv2.waitKey(1)
                    return frame
                return None
                
            def instructions_callback(message):
                print(f"Challenge: {message}")
                
            # Run challenge-response test
            result = liveness_detector.run_challenge_response_test(
                capture_callback,
                instructions_callback
            )
            
            print(f"Challenge-response test: {'Passed' if result else 'Failed'}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()