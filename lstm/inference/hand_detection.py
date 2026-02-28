import mediapipe as mp

# Used to convert protobuf message
# to a dictionary.


from google.protobuf.json_format import MessageToDict


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.8,
    max_num_hands=2)







