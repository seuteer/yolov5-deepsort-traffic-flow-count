import cv2


def select_polygons(video_file):
    def draw_polygon(event, x, y, flags, param):
        nonlocal point_list, num_polygons, img, reset
        if event == cv2.EVENT_LBUTTONDOWN:
            if reset:
                point_list = [[] for _ in range(2)]
                num_polygons = 0
                reset = False
            if len(point_list[num_polygons]) < 4:
                point_list[num_polygons].append((x, y))
                if len(point_list[num_polygons]) == 1:
                    # Draw the vertices as circles
                    cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                else:
                    # Draw the edges as lines
                    cv2.line(img, point_list[num_polygons][-2], (x, y), (0, 255, 0), 2)
                if len(point_list[num_polygons]) == 4:
                    # Draw the final edge connecting the last and first vertices
                    cv2.line(img, point_list[num_polygons][-1], point_list[num_polygons][0], (0, 255, 0), 2)
                    num_polygons += 1

    # 读取视频的第一帧
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1920, 1080))

    if ret:
        img = frame.copy()

        # 创建两个多边形框的坐标列表
        point_list = [[] for _ in range(2)]
        num_polygons = 0
        reset = False

        # 显示第一帧图像，并在图像上绘制多边形框
        cv2.namedWindow('Select Polygons')
        cv2.setMouseCallback('Select Polygons', draw_polygon)

        while True:
            cv2.imshow('Select Polygons', img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or num_polygons >= 2:
                break
            elif key == ord('r'):
                reset = True
                img = frame.copy()

        cv2.destroyAllWindows()

        # 返回两个多边形框的坐标列表
        polygons_coordinates = point_list[:2]
        return polygons_coordinates
    else:
        print("Failed to read the video.")
        return None
