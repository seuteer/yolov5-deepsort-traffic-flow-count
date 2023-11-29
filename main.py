import cv2
import numpy as np
import random
import tracker
import draw_polygon
from detector import Detector
import pandas as pd


if __name__ == '__main__':

    # 按车流方向分别绘制两个多边形框，区分下行和上行方向
    video_file = './video/交叉口-5min-3x.mp4'
    polygon_mask_dict = {}
    color_polygons_image_dict = {}
    dict_overlapping_1_polygon = {}
    dict_overlapping_2_polygon = {}
    dict_overlapping_all = {}
    down_up_dict = {}
    flow_colors = [
        (255, 0, 0),  # 红色
        (0, 255, 0),  # 绿色
        (0, 0, 255),  # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
    ]

    # 绘制多条流向，统计各个流向的车流量
    flow_list = []
    while True:
        # 输入流向（输入为空，跳出循环）
        flow = input('FLOW（E/W/S/N，S/L/R）: ')
        if flow:
            flow_list.append(flow)
        else:
            break

        # 初始化2个撞线polygon
        list_pts = draw_polygon.select_polygons(video_file)

        # 根据视频尺寸，填充一个polygon，供撞线计算使用
        mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
        list_pts_1 = list_pts[0]
        ndarray_pts_1 = np.array(list_pts_1, np.int32)
        polygon_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_1], color=1)
        polygon_value_1 = polygon_value_1[:, :, np.newaxis]
        # 填充第二个polygon
        mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
        list_pts_2 = list_pts[1]
        ndarray_pts_2 = np.array(list_pts_2, np.int32)
        polygon_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_2], color=2)
        polygon_value_2 = polygon_value_2[:, :, np.newaxis]

        # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
        polygon_mask = polygon_value_1 + polygon_value_2
        # 缩小尺寸，1920x1080->960x540
        polygon_mask = cv2.resize(polygon_mask, (960, 540))
        polygon_mask_dict[flow] = polygon_mask

        image_1 = np.array(polygon_value_1 * flow_colors[len(flow_list)-1], np.uint8)
        image_2 = np.array(polygon_value_2 * flow_colors[len(flow_list)-1], np.uint8)

        # 彩色图片（值范围 0-255）
        color_polygons_image = image_1 + image_2
        # 缩小尺寸，1920x1080->960x540
        color_polygons_image = cv2.resize(color_polygons_image, (960, 540))
        color_polygons_image_dict[flow] = color_polygons_image

        dict_overlapping_1_polygon[flow] = []
        dict_overlapping_2_polygon[flow] = []

        # 存储上下行每个检测类别总数
        down_up_dict[flow + '_down'] = {'total': 0}
        down_up_dict[flow + '_up'] = {'total': 0}

    # 初始化 yolov5
    detector = Detector()

    # 打开视频
    capture = cv2.VideoCapture(video_file)

    while True:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        # 目标检测，生成列表存放检测物体
        list_bboxs = []
        bboxes = detector.detect(im)

        # 如果画面中 有bbox
        if len(bboxes) > 0:
            list_bboxs = tracker.update(bboxes, im)

            # 画框
            # 撞线检测点，检测框中心位置
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=1)
            pass
        else:
            # 如果画面中 没有bbox
            output_image_frame = im
        pass

        # 输出图片
        for flow in flow_list:
            output_image_frame = cv2.add(output_image_frame, color_polygons_image_dict[flow])

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for flow in flow_list:
                for item_bbox in list_bboxs:
                    x1, y1, x2, y2, label, track_id = item_bbox

                    # 撞线的点(计算矩形框的中心点 x 坐标)
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)

                    if polygon_mask_dict[flow][y, x] == 1:
                        # 如果撞 polygon1
                        if track_id not in dict_overlapping_1_polygon[flow]:
                            dict_overlapping_1_polygon[flow].append(track_id)
                        pass

                        # 判断 polygon2 list 里是否有此 track_id
                        # 有此 track_id，则 认为是 外出方向
                        if track_id in dict_overlapping_2_polygon[flow]:
                            # 外出+1
                            if label not in down_up_dict[flow + '_up']:
                                down_up_dict[flow + '_up'][label] = 1
                            else:
                                down_up_dict[flow + '_up'][label] += 1
                            down_up_dict[flow + '_up']['total'] += 1
                            print(
                                f"{flow} | 类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {down_up_dict[flow + '_up']['total']} | 上行id列表: {dict_overlapping_2_polygon[flow]}")

                            # 删除 polygon2 list 中的此id
                            dict_overlapping_2_polygon[flow].remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass

                    elif polygon_mask_dict[flow][y, x] == 2:
                        # 如果撞 polygon2
                        if track_id not in dict_overlapping_2_polygon[flow]:
                            dict_overlapping_2_polygon[flow].append(track_id)
                        pass

                        # 判断 polygon1 list 里是否有此 track_id
                        # 有此 track_id，则 认为是 进入方向
                        if track_id in dict_overlapping_1_polygon[flow]:
                            # 进入+1
                            if label not in down_up_dict[flow + '_down']:
                                down_up_dict[flow + '_down'][label] = 1
                            else:
                                down_up_dict[flow + '_down'][label] += 1
                            down_up_dict[flow + '_down']['total'] += 1
                            print(
                                f"{flow} | 类别: {label} | id: {track_id} | 下行撞线 | 下行撞线总数: {down_up_dict[flow + '_down']['total']} | 下行id列表: {dict_overlapping_1_polygon[flow]}")

                            # 删除 蓝polygon list 中的此id
                            dict_overlapping_1_polygon[flow].remove(track_id)

                            pass
                        else:
                            # 无此 track_id，不做其他操作
                            pass
                        pass
                    else:
                        pass
                    pass

                pass

                # # ----------------------清除无用id----------------------
                # dict_overlapping_all[flow] = dict_overlapping_1_polygon[flow] + dict_overlapping_2_polygon[flow]
                # for id1 in dict_overlapping_all[flow]:
                #     is_found = False
                #     for _, _, _, _, _, bbox_id in list_bboxs:
                #         if bbox_id == id1:
                #             is_found = True
                #             break
                #         pass
                #     pass
                #
                #     if not is_found:
                #         # 如果没找到，删除id
                #         if id1 in dict_overlapping_2_polygon[flow]:
                #             dict_overlapping_2_polygon[flow].remove(id1)
                #         pass
                #         if id1 in dict_overlapping_1_polygon[flow]:
                #             dict_overlapping_1_polygon[flow].remove(id1)
                #         pass
                #     pass
                # dict_overlapping_all[flow].clear()
                # pass
                # # ----------------------清除无用id----------------------

            # 清空list
            list_bboxs.clear()
            pass

        else:
            # 如果图像中没有任何的bbox，则清空list
            for flow in flow_list:
                dict_overlapping_1_polygon[flow].clear()
                dict_overlapping_2_polygon[flow].clear()
            pass
        pass

        # 文字位置
        font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
        draw_text_postion = [int(960 * 0.01), int(540 * 0.05)]
        for ind, flow in enumerate(flow_list):
            text_draw = flow + ' DOWN:' + str(down_up_dict[flow + '_down']['total']) + ', UP:' + str(down_up_dict[flow + '_up']['total'])
            output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                             org=draw_text_postion,
                                             fontFace=font_draw_number,
                                             fontScale=1, color=flow_colors[ind], thickness=2)
            draw_text_postion[1] += 30

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass

    print(flow, down_up_dict)
    pd.DataFrame(down_up_dict).T.to_excel(video_file.replace('.mp4', '.xlsx'))
    capture.release()
    cv2.destroyAllWindows()
