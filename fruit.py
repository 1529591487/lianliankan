import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import random

from PIL import ImageGrab
import numpy as np

from collections import deque
import matplotlib.pyplot as plt

# 定义方向向量，上、下、左、右
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

DEBUG = 0

SHOW_SCALE = 1
         #1 2  3   4   5    6   7   8   9   10  11  12  13  14
x_list = [3,57,111,165,219,273,327,382,436,490,544,598,652,706]
y_list = [2,52,101,150,199,249,298,347,396,446]

def check_empty(block,x,y,w,h):
    blue_channel = block[:, :, 0]  # 获取蓝色通道
    green_channel = block[:, :, 1]  # 获取绿色通道
    red_channel = block[:, :, 2]    # 获取红色通道

    # 计算每个通道的均值
    mean_blue = np.mean(blue_channel)
    mean_green = np.mean(green_channel)
    mean_red = np.mean(red_channel)

    # print(mean_blue,mean_green,mean_red)

    # 设置蓝色背景的阈值，可以根据你的图像调整
    blue_threshold = 230
    green_threshold = 200
    red_threshold = 220

    # 如果蓝色通道占主导地位，且绿色和红色通道都较低，则判断为空
    if mean_blue > blue_threshold and mean_green > green_threshold and mean_red < red_threshold:
        print(f"Block at ({x}, {y}, {w}, {h}) is empty (blue background).")
        # 用红色矩形标记空块
        return True
    else:
        return False

def preprocess_image_by_list(img):
    img_h, img_w, _ = img.shape
    blocks = []

    # img_copy = img.copy()
    for i in range(0,len(x_list)):
        for j in range(0,len(y_list)):
            x = x_list[i] + 1
            y = y_list[j] + 1
            w = 47
            h = 43
            block = img[y:(y+h), x:(x+w)]
            if True == check_empty(block,x,y,w,h):
                continue

            blocks.append((block, (x, y,w, h)))
            cv2.rectangle(img, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 3)

    if DEBUG:
        window_name = 'Detected Blocks'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 调整窗口大小，例如宽度400，高度300
        cv2.resizeWindow(window_name, int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return blocks    

def preprocess_image(img, grid_size=(10, 14)):
    img_h, img_w, _ = img.shape

    # 读取图片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转为灰度图
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯模糊，减少噪声
    edges = cv2.Canny(gray, 4, 4)  # 使用 Canny 边缘检测

    # 显示边缘检测结果
    if DEBUG:
        window_name = 'Edges with Canny'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 调整窗口大小，例如宽度400，高度300
        cv2.resizeWindow(window_name, int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
        cv2.imshow(window_name, edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 使用形态学操作来填充边缘空隙
    # 使用较小的核进行形态学操作
    # kernel = np.ones((2, 2), np.uint8)  # 使用较小的核
    # edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)  # 使用闭操作代替单纯膨胀

    # 显示边缘检测结果
    if DEBUG:
        window_name = 'Edges with Morphology'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 调整窗口大小，例如宽度400，高度300
        cv2.resizeWindow(window_name, int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
        cv2.imshow(window_name, edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 查找图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []  # 存储分割出来的每个方块和其坐标
    board_size = 7

    min_block_height = img_h // 2 // grid_size[0]  # 使用整除，确保尺寸为整数
    min_block_width = img_w // 2 // grid_size[1]

    max_block_height = img_h // grid_size[0] + 5  # 使用整除，确保尺寸为整数
    max_block_width = img_w // grid_size[1] + 5

    x_dict = {}
    y_dict = {}
    # 遍历检测到的每个轮廓
    for contour in contours:
        # 为每个轮廓拟合一个最小的边界矩形
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤掉太小或太大的矩形块，防止噪声干扰
        if w > min_block_width and w < max_block_width and h > min_block_height and h < max_block_height:  # 根据图像大小调整阈值
            # 提取矩形区域
            block = img[y:y+h, x:x+w]
            blocks.append((block, (x, y, w, h)))

            # 在原图上绘制矩形框 (绿色)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            x_dict.update({x:1})
            y_dict.update({y:1})

            # 按照键排序并输出
    for key in sorted(x_dict):
        print(key,end=",")
    print("")
            # 按照键排序并输出
    for key in sorted(y_dict):
        print(key,end=",")
    print("")

    if DEBUG:
        window_name = 'Detected Blocks'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 调整窗口大小，例如宽度400，高度300
        cv2.resizeWindow(window_name, int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
        cv2.imshow(window_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return blocks

def orb_similarity(block1, block2):
    # 初始化 ORB 检测器
    orb = cv2.ORB_create()
    
    # 提取关键点和描述符
    kp1, des1 = orb.detectAndCompute(block1, None)
    kp2, des2 = orb.detectAndCompute(block2, None)

    # 使用暴力匹配器进行匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 根据距离排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 返回相似度：匹配点数与关键点数之比
    similarity = len(matches) / max(len(kp1), len(kp2))
    return similarity

def match_blocks_by_classification(blocks, threshold=0.7,resize = 15):
    classified_blocks = []  # 存储分类后的块
    unclassified_blocks = blocks.copy()  # 复制一个待分类的列表

    while unclassified_blocks:
        block1, pos1 = unclassified_blocks[0]  # 选择第一个未分类的块
        current_class = [(block1, pos1)]  # 当前分类的块
        remaining_blocks = []

        # 与剩余的块比较
        for i in range(1, len(unclassified_blocks)):
            block2, pos2 = unclassified_blocks[i]

            # 将两个图像转换为灰度图，以便于计算 SSIM
            block1_gray = cv2.cvtColor(block1, cv2.COLOR_BGR2GRAY)
            block2_gray = cv2.cvtColor(block2, cv2.COLOR_BGR2GRAY)

            # 调整大小以加快计算（可选）
            block1_resized = cv2.resize(block1_gray, (resize, resize))
            block2_resized = cv2.resize(block2_gray, (resize, resize))

            # 计算 SSIM（结构相似性）
            score, _ = ssim(block1_resized, block2_resized, full=True)

            # 如果相似性大于设定的阈值，则将其归入同一类
            if score > threshold:
                current_class.append((block2, pos2))
            else:
                remaining_blocks.append((block2, pos2))

        classified_blocks.append(current_class)
        unclassified_blocks = remaining_blocks

    return classified_blocks

def draw_classified_blocks(image, classified_blocks):
    colors = {}  # 存储每个分类的颜色
    image_copy = image.copy()

    # 为每个分类生成随机颜色，并绘制矩形
    for class_idx, block_class in enumerate(classified_blocks):
        # 为每个类别选择一种随机颜色
        color = tuple([random.randint(0, 255) for _ in range(3)])
        colors[class_idx] = color

        # 遍历每个类中的所有块，绘制矩形
        for _, (x, y, w, h) in block_class:
            cv2.rectangle(image_copy, (x, y), (x + w, y + h), color, 3)

    img_h, img_w, _ = image.shape
    # 显示带分类标记的图像
    # 创建窗口，默认不可以调整大小
    if 1:
        window_name = 'Classified Blocks'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 调整窗口大小，例如宽度400，高度300
        cv2.resizeWindow(window_name, int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
        cv2.imshow(window_name, image_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def is_valid(x, y, grid, visited):
    """ 判断坐标 (x, y) 是否是合法的网格内坐标 """
    rows, cols = len(grid), len(grid[0])
    return 0 <= x < rows and 0 <= y < cols and not visited[x][y] and grid[x][y] == 0

def bfs_find_path(start, end, grid):
    """ 使用广度优先搜索 (BFS) 来寻找两个点之间的可行路径 """
    extened_start = (start[0]+1,start[1]+1)
    extened_end = (end[0]+1,end[1]+1)
    queue = deque([(extened_start, -1, 1, [extened_start])])  # (位置, 前一个方向, 拐点数量, 当前路径)
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    visited[extened_start[0]][extened_start[1]] = True

    print(f"start:{extened_start} end:{extened_end}")

    while queue:
        (x, y), prev_dir, bends, path = queue.popleft()

        # 调试信息：打印当前节点位置
        print(f"Visiting node: {(x, y)}, bends: {bends}, path length: {len(path)}")

        # 如果到达终点，返回路径
        if start == (6,1) and end == (3,1):
            if x == 3 and y == 0:
                a = 0
        if (x, y) == extened_end:
            print(f"Path found: {path}")
            return path

        # 尝试四个方向
        for i, (dx, dy) in enumerate(DIRECTIONS):
            nx, ny = x + dx, y + dy
            print(f"node: {(nx, ny)}, bends: {bends}, path length: {len(path)}")
            if (nx, ny) == extened_end:
                print(f"Path found: {path}")
                path = path + [(nx, ny)]
                return path
            if is_valid(nx, ny, grid, visited):
                new_bends = bends
                if prev_dir != -1 and prev_dir != i:
                    new_bends += 1  # 如果方向改变了，增加一个拐点

                # 如果拐点数不超过 3，继续搜索
                if new_bends <= 2:
                    visited[nx][ny] = True
                    queue.append(((nx, ny), i, new_bends, path + [(nx, ny)]))

    print("No path found")
    return None  # 如果找不到路径，返回 None

def draw_path(image,pos1,pos2, path, grid_size, extened_grid_size, color=(255, 0, 0)):
    """ 在图像上绘制路径 """
    img_h, img_w = image.shape[:2]
    cell_h, cell_w = img_h // extened_grid_size[0], img_w // extened_grid_size[1]

    cv2.rectangle(image, (pos1[0], pos1[1]), (pos1[0] + pos1[2], pos1[1] + pos1[3]), (0, 0, 255), 5)
    cv2.rectangle(image, (pos2[0], pos2[1]), (pos2[0] + pos2[2], pos2[1] + pos2[3]), (0, 0, 255), 5)

    for i in range(1, len(path)):
        start = ((path[i-1][1]) * cell_w + cell_w // 2, (path[i-1][0]) * cell_h + cell_h // 2)
        end = ((path[i][1]) * cell_w + cell_w // 2, (path[i][0]) * cell_h + cell_h // 2)
        cv2.line(image, start, end, color, 5)

    cv2.namedWindow('path', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('path', int(img_w * SHOW_SCALE) , int(img_h * SHOW_SCALE))
    cv2.imshow('path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_grid(image, classified_blocks, grid_size,extened_grid_size):
    """ 将图像块转换为网格表示，空白位置为0，方块位置为1 """
    # grid = np.zeros(grid_size, dtype=int)
    grid = np.zeros(extened_grid_size, dtype=int)

    # 遍历所有分类后的方块
    for block_class in classified_blocks:
        for _, (x, y, w, h) in block_class:  # 遍历每个分类中的块
            # 计算块的中心位置，并将其映射到网格坐标
            center_x = x + w // 2
            center_y = y + h // 2

            grid_y = center_y // (image.shape[0] // grid_size[0]) + 1
            grid_x = center_x // (image.shape[1] // grid_size[1]) + 1

            # 确保坐标在网格范围内
            if 0 <= grid_x < extened_grid_size[1] and 0 <= grid_y < extened_grid_size[0]:
                grid[grid_y][grid_x] = 1  # 将该位置标记为1（表示有方块）

    return grid


def display_grid(grid):
    plt.imshow(grid, cmap='gray')
    plt.title('Generated Grid')
    plt.show()


def find_and_draw_paths(image, classified_blocks, grid_size=(10, 14)):
    """ 找到每对相同方块的匹配路径，并在图像上绘制 """
    extened_grid_size = (grid_size[0]+2,grid_size[1]+2)
    grid = generate_grid(image, classified_blocks, grid_size, extened_grid_size)
    if DEBUG:
        display_grid(grid)

    height, width = image.shape[:2]

    # 计算网格单元的大小
    grid_height = int(height / 10)
    grid_width = int(width / 14)

    # 在图像四周添加一个网格单元的白边
    border_size = (grid_width, grid_height)  # 左右的宽度、上下的高度

    # 添加白边
    image_with_border = cv2.copyMakeBorder(
        image,
        border_size[1], border_size[1], border_size[0], border_size[0],
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255]
    )

    # # 显示结果
    # cv2.imshow('Image with Border', image_with_border)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for block_class in classified_blocks:
        if len(block_class) < 2:
            continue  # 如果某类方块小于两个，跳过

        for i in range(len(block_class)):
            for j in range(i+1, len(block_class)):
                _, pos1 = block_class[i]
                _, pos2 = block_class[j]

                start_center_x = (pos1[0] + pos1[2] // 2)
                start_center_y = (pos1[1] + pos1[3] // 2)

                end_center_x = (pos2[0] + pos2[2] // 2)
                end_center_y = (pos2[1] + pos2[3] // 2)

                # 将图像坐标转换为网格坐标
                start_x = (pos1[0] + pos1[2] // 2) // (image.shape[1] // grid_size[1])
                start_y = (pos1[1] + pos1[3] // 2) // (image.shape[0] // grid_size[0])
                end_x = (pos2[0] + pos2[2] // 2)  // (image.shape[1] // grid_size[1])
                end_y = (pos2[1] + pos2[3] // 2) // (image.shape[0] // grid_size[0])

                start = (start_y, start_x)
                end = (end_y, end_x)

                # 检查是否在网格范围内
                if 0 <= start[0] < grid_size[0] and 0 <= start[1] < grid_size[1] and \
                   0 <= end[0] < grid_size[0] and 0 <= end[1] < grid_size[1]:

                    if 0 == start_x and 2 == start_y:
                        a = 0

                    if 0 == start_x and 5 == start_y:
                        a = 0

                    # 通过 BFS 寻找路径
                    path = bfs_find_path(start, end, grid)

                    # 如果找到路径，则绘制
                    if path:
                        # image_copy = image.copy()
                        # cv2.rectangle(image_copy, (pos1[0], pos1[1]), (pos1[0] + pos1[2], pos1[1] + pos1[3]), (0, 0, 255), 5)
                        # cv2.rectangle(image_copy, (pos2[0], pos2[1]), (pos2[0] + pos2[2], pos2[1] + pos2[3]), (0, 0, 255), 5)

                        # cv2.namedWindow('Block Centers', cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow('Block Centers', image.shape[1] // 2, image.shape[0] // 2)
                        # cv2.imshow('Block Centers', image_copy)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        extened_pos1 = []
                        for item in pos1:
                            extened_pos1.append(item)
                        extened_pos1[0] += border_size[0]
                        extened_pos1[1] += border_size[1]

                        extened_pos2 = []
                        for item in pos2:
                            extened_pos2.append(item)
                        extened_pos2[0] += border_size[0]
                        extened_pos2[1] += border_size[1]
                        image_copy = image_with_border.copy()
                        draw_path(image_copy, extened_pos1,extened_pos2,path, grid_size , extened_grid_size)
                else:
                    print(f"Warning: start or end point out of bounds: start={start}, end={end}")

def main(image_path):

    if 0:
        screenshot_cv2 = cv2.imread(image_path)
    else:

        # 定义要截取的区域 (left, top, right, bottom)
        # bbox = (460, 546, 1752, 1394)  # 根据需要调整坐标
        bbox = (93, 299, 850, 792)  # 根据需要调整坐标

        # 截取指定区域
        screenshot = ImageGrab.grab(bbox=bbox)

        # 将截图转换为 NumPy 数组格式
        screenshot_np = np.array(screenshot)

        # 将图像从 RGB 转换为 BGR 以符合 OpenCV 的格式
        screenshot_cv2 = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)


    # 步骤 1：图片预处理
    blocks = preprocess_image_by_list(screenshot_cv2)

    # 步骤 2：方块匹配
    classified_blocks = match_blocks_by_classification(blocks, threshold=0.9,resize = 20)
    draw_classified_blocks(screenshot_cv2, classified_blocks)

    # 寻找并绘制路径
    find_and_draw_paths(screenshot_cv2, classified_blocks, grid_size=(10, 14))

main("input1.jpg")