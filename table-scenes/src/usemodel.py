from ultralytics import YOLOE

# Initialize a YOLOE model
model = YOLOE("yoloe-11l-poker-memory.onnx")  # or select yoloe-11s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
names = ["person", "poker", "chopsticks"]
model.set_classes(names, model.get_text_pe(names))

# Run detection on the given image
results = model.predict(source=1, conf=0.3)  # Use 0.3 confidence threshold


# Show results
results[0].show()