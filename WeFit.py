import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import numpy as np
from ComparePose import compare_pose

text="Correct"

class WebcamApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Webcam Viewer")
        
        self.cap = None
        self.video_label = tk.Label(self.master)
        self.video_label.pack()
        self.counter=0
        self.create_buttons()
    
    def create_buttons(self):
        asanas = ["Taarasana", "Natrajasana", "Trikonasana", "Vrikshasana", "Padmasana"]
        for name in asanas:
            button = tk.Button(self.master, text=name, command=lambda n=name: self.start_video_stream(n))
            button.pack(pady=10)
    
    def start_video_stream(self, asana_name):
        if self.cap is not None:
            self.stop_video_stream()  # Stop current video stream
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return
        
        self.show_frame(asana_name)
    
    def stop_video_stream(self):
        if self.cap is not None:
            self.cap.release()
            self.video_label.config(image=None)  # Clear the image label
            
    
    def show_frame(self, asana_name):
        global text
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = img.resize((640, 480), Image.LANCZOS)  # Use LANCZOS filter for resizing
            self.counter+=1
            if self.counter>=100:
                np_image = np.array(img)
                
                val=compare_pose(asana_name,np_image)
                if val:
                    text="Correct"
                else:
                    text="Incorrect"
                self.counter=0
                
            draw = ImageDraw.Draw(img)
            font_size = 30
            font_color = (255, 255, 255)  # White color in RGB format
            font_path = "arial.ttf"  # Path to the font file (change as needed)

            # Load the font
            font = ImageFont.truetype(font_path, font_size)

            # Calculate text position (top-left corner)
            # text_width, text_height = draw.textsize(text, font)
            text_x = 10  # Adjust as needed
            text_y = 10  # Adjust as needed

            # Draw text on the image
            draw.text((text_x, text_y), text, font=font, fill=font_color)
            
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.video_label.img = img_tk  # Keep a reference to prevent garbage collection
            self.video_label.config(image=img_tk)
            self.video_label.after(10, lambda: self.show_frame(asana_name))  # Update every 10 milliseconds
            
            # Press 'q' to stop the video stream
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_video_stream()
                cv2.destroyAllWindows()
                
        else:
            print("Error: Failed to capture frame.")
    
    def compare_pose(image):
        pass

def main():
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
