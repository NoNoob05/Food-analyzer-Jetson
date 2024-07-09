import jetson.inference
import jetson.utils

net = jetson.inference.detectNet(argv=['--model=frutas.onnx', '--labels=labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])

camera=jetson.utils.videoSource("/dev/video0")
display=jetson.utils.videoOutput()

while True:
    img=camera.Capture()
    detect=net.Detect(img)

    if detect:
        print("DETECTION FRUIT")

    display.Render(img)
    display.SetStatus("Fruits | Network {:.0f}FPS".format(net.GetNetworkFPS()))

    if not camera.IsStreaming() or not display.IsStreaming():
        break
