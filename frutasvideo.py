import flask
import sys
import os

app = flask.Flask(__name__)

@app.route('/')
def home():
    return "Food Analyzer Jetson"

if __name__ == "__main__":
    # Check if running on a Jetson device
    ON_JETSON = False
    try:
        import jetson.inference
        import jetson.utils
        ON_JETSON = True
    except ImportError:
        # Mock the jetson modules if not running on a Jetson device
        class MockJetson:
            def Detect(self, img):
                return []

            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        jetson = MockJetson()
        jetson.utils = MockJetson()

    if ON_JETSON:
        net = jetson.inference.detectNet(argv=['--model=frutas.onnx', '--labels=labels.txt', '--input-blob=input_0', '--output-cvg=scores', '--output-bbox=boxes'])
        camera = jetson.utils.videoSource("/dev/video0")
        display = jetson.utils.videoOutput()
    else:
        net = MockJetson()
        camera = MockJetson()
        display = MockJetson()

    while True:
        if ON_JETSON:
            img = camera.Capture()
        else:
            img = None  # Mock image or handle accordingly

        detect = net.Detect(img)
        if detect:
            print("DETECTION FRUIT")

        if ON_JETSON:
            display.Render(img)
            display.SetStatus("Fruits | Network {:.0f}FPS".format(net.GetNetworkFPS()))

        if not ON_JETSON or not camera.IsStreaming() or not display.IsStreaming():
            break

    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
