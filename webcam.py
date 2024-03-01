import cv2
import keyboard as kb
import time

def main():
    vid = cv2.VideoCapture(0) 

    #while(not kb.is_pressed('f4')): 
    while(True):
        ret, frame = vid.read() 
  
        # Display the resulting frame 
        cv2.imshow('frame', frame) 

        # Break if key detected
        if cv2.waitKey(1) & 0xFF == 240: #make it equal something it cant
            break
    # After the loop release the cap object 
    vid.release()
    # Destroy all the windows 
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    main()