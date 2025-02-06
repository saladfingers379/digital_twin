from ultralytics import YOLO
import cv2
from vpython import canvas, box, vector, color, label, scene, rate

def iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) for two bounding boxes.
    Each box is a tuple (x1, y1, x2, y2).
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea == 0 or boxBArea == 0:
        return 0.0
    return interArea / float(boxAArea + boxBArea - interArea)

def main():
    # -----------------------------------
    # Configuration
    # -----------------------------------
    model_path = r"C:\Users\joshu\Documents\EEE\AMNIS\Models\training\weights\best.pt"  # update as needed
    video_path = r"C:\Users\joshu\Queen's University Belfast\Michael Loughran - 20241213_mb_bad\20241213_113344_BR4.mp4"  # update as needed
    detection_conf_thresh = 0.2   # YOLO confidence threshold
    detection_nms_thresh = 0.2    # (optional) NMS threshold
    shelf_thresh = 100            # Pixel difference to group boxes into the same shelf

    # Scale factor to convert from pixels to VPython (3D) units.
    # (Adjust this factor to set the overall size of your digital twin.)
    scale = 0.01

    # -----------------------------------
    # 1. Get first frame and ROI selection
    # -----------------------------------
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read the first frame of the video.")
    cap.release()

    # Let the user select the shelf ROI (this window will pause until ROI is selected)
    print("Select ROI and press ENTER. Press C to cancel.")
    roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")
    x_min, y_min, w, h = roi
    x_max, y_max = x_min + w, y_min + h

    # For our digital twin, the ROI dimensions will become our “shelf area.”
    ROI_width = w
    ROI_height = h

    # -----------------------------------
    # 2. Run YOLO detection on the first frame
    # -----------------------------------
    model = YOLO(model_path)
    results = model(first_frame, conf=detection_conf_thresh, iou=detection_nms_thresh)[0]

    # Filter boxes to only those within the ROI
    boxes = []
    for b in results.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        if (x_min <= x1 <= x_max and x_min <= x2 <= x_max and
            y_min <= y1 <= y_max and y_min <= y2 <= y_max):
            boxes.append((x1, y1, x2, y2))

    # -----------------------------------
    # 3. Group boxes into shelves (rows) based on vertical positions
    # -----------------------------------
    boxes.sort(key=lambda b: b[3])  # sort by the bottom coordinate
    shelves = []  # each shelf will be a list of boxes
    for bx in boxes:
        placed = False
        for shelf in shelves:
            shelf_mean_bottom = sum(b[3] for b in shelf) / len(shelf)
            if abs(bx[3] - shelf_mean_bottom) <= shelf_thresh:
                shelf.append(bx)
                placed = True
                break
        if not placed:
            shelves.append([bx])

    # Also sort boxes within each shelf left-to-right
    for shelf in shelves:
        shelf.sort(key=lambda b: b[0])

    # Create active objects with unique IDs (for labeling in 3D)
    active_objects = [(i+1, bx) for i, bx in enumerate(boxes)]

    # -----------------------------------
    # 4. Create a 3D digital twin using VPython
    # -----------------------------------
    # Set up the VPython canvas.
    scene = canvas(title="Digital Twin 3D Shelf", width=800, height=600,
                   center=vector((ROI_width/2)*scale, (ROI_height/2)*scale, 0))
    scene.background = color.gray(0.2)

    # For each shelf group, create a shelf board.
    # We assume each shelf board spans the full ROI width.
    shelf_thickness = 0.2   # thickness of the shelf board in 3D units
    shelf_depth = 1         # how deep the shelf is
    shelf_boards = []

    # We map the image’s vertical coordinate (with y=0 at the top) to VPython’s y-axis (increasing upward).
    # Here, a point at image y (relative to ROI) becomes: digital_y = (ROI_height - y)*scale.
    for shelf in shelves:
        # Compute a representative y position for the shelf.
        shelf_y_pixels = min(b[3] for b in shelf) - y_min  # relative y (0 = top of ROI)
        digital_y = (ROI_height - shelf_y_pixels) * scale
        digital_x = (ROI_width/2) * scale  # center of shelf (horizontally)
        shelf_board = box(pos=vector(digital_x, digital_y - shelf_thickness/2, 0),
                          size=vector(ROI_width*scale, shelf_thickness, shelf_depth),
                          color=color.cyan,
                          opacity=0.5)
        shelf_boards.append(shelf_board)

    # For each detected object, create a 3D representation (a box) and a label.
    for (obj_id, (x1, y1, x2, y2)) in active_objects:
        # Compute the center of the bounding box relative to the ROI.
        center_x = ((x1 + x2) / 2) - x_min
        center_y = ((y1 + y2) / 2) - y_min
        digital_x = center_x * scale

        # Determine which shelf this object belongs to by checking which shelf group contains the box.
        assigned_shelf = None
        for shelf in shelves:
            if (x1, y1, x2, y2) in shelf:
                assigned_shelf = shelf
                break
        if assigned_shelf is not None:
            shelf_index = shelves.index(assigned_shelf)
            shelf_board = shelf_boards[shelf_index]
            # Place the object on top of the shelf board.
            shelf_top = shelf_board.pos.y + shelf_thickness/2
            # Compute the object's size (scaled).
            obj_width = (x2 - x1) * scale
            obj_height = (y2 - y1) * scale
            digital_y = shelf_top + obj_height/2  # so the object sits on the shelf
        else:
            # Fallback: place the object based on its digital y value.
            digital_y = (ROI_height - center_y) * scale

        # For simplicity, we place all objects at z=0 (you could vary this for depth effects)
        digital_z = 0

        # Create the 3D box representing the object.
        obj_box = box(pos=vector(digital_x, digital_y, digital_z),
                      size=vector((x2 - x1) * scale, (y2 - y1) * scale, 0.5),
                      color=color.orange)

        # Add a label with the object ID
        label(pos=obj_box.pos, text=str(obj_id), xoffset=0, yoffset=10,
              height=10, color=color.black, box=False)

    # Add an overall title label in the scene.
    label(pos=vector((ROI_width*scale)/2, ROI_height*scale + 0.5, 0),
          text="Digital Twin of Shelf", xoffset=0, yoffset=20, height=16,
          color=color.white, box=False)

    print("3D digital twin rendered. You can rotate/zoom the scene.")

    # -----------------------------------------------------------
    # Prevent the script from exiting so the VPython canvas remains.
    # -----------------------------------------------------------
    while True:
        rate(10)  # this loop keeps the program alive at 10 iterations per second

if __name__ == "__main__":
    main()
