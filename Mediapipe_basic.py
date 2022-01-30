import cv2
import numpy as np
import mediapipe as mp

# mediapipe入门

# 画图必备
mp_drawing = mp.solutions.drawing_utils
# 默认绘图风格
mp_drawing_styles = mp.solutions.drawing_styles
# 自定义绘图风格   参数：1、颜色，2、线条粗细，3、点的半径
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 1, 1)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 1, 1)

# 导入方法
mp_face_detection = mp.solutions.face_detection  # 人脸检测
mp_face_mesh = mp.solutions.face_mesh  # 面网格
mp_hand = mp.solutions.hands  # 手
mp_pose = mp.solutions.pose  # 姿势

# 参数设置，使用"with 等号右侧 as 等号左侧:“，简便且安全
mode_face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
mode_face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                       min_detection_confidence=0.5)
mode_hand = mp_hand.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mode_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
'''
with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as mode_face_detection:
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as mode_face_mesh:
        with mp_hand.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as mode_hand:
            with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                              min_detection_confidence=0.5) as mode_pose:
'''
# 开启摄像头
cap = cv2.VideoCapture(0)
# 判断开启摄像头是否成功
while cap.isOpened():
    success, img = cap.read()
    # 玩一玩，打开摄像头失败时弹出图片
    blank = np.zeros((720, 1280, 3))
    if not success:
        cv2.putText(blank, 'Please Check Your Camera', (150, 250), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow('Text', blank)
        cv2.waitKey(1)
        continue

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 增强对于视觉的识别能力

    # RGB图像处理
    result_face_detection = mode_face_detection.process(img)
    result_face_mesh = mode_face_mesh.process(img)
    result_hand = mode_hand.process(img)
    result_pose = mode_pose.process(img)

    # 绘制关键点与连线
    img.flags.writeable = True  # 官方文件都有这个，还不理解作用是啥
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 恢复原图像色彩
    # 绘制脸部识别
    if result_face_detection.detections:
        for detection in result_face_detection.detections:
            mp_drawing.draw_detection(img, detection)
    # 绘制脸部网孔
    if result_face_mesh.multi_face_landmarks:
        for face_landmarks in result_face_mesh.multi_face_landmarks:

            mp_drawing.draw_landmarks(image=blank, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_TESSELATION,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_tesselation_style())

            mp_drawing.draw_landmarks(image=blank, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_CONTOURS,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_contours_style())

            mp_drawing.draw_landmarks(image=blank, landmark_list=face_landmarks,
                                      connections=mp_face_mesh.FACEMESH_IRISES,
                                      connection_drawing_spec=mp_drawing_styles
                                      .get_default_face_mesh_iris_connections_style())


    # 绘制手部节点
    if result_hand.multi_hand_landmarks:
        for hand_landmarks in result_hand.multi_hand_landmarks:
            mp_drawing.draw_landmarks(blank, hand_landmarks, mp_hand.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    # 绘制身体节点

    mp_drawing.draw_landmarks(blank, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # 翻转镜头，设置关闭条件
    cv2.imshow('MediaPipe Face Detection', cv2.flip(blank, 1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
