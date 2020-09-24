import cv2
import numpy as np

def combine_mask_to_image(src_img_path, dst_img_path, alpha_thold=(230, 230, 230),begin_pixel=(0 , 0)):
    """
    :param src_img_path: 需要贴上去的图片
    :param dst_img_path: 背景图片
    :begin_pixel: 开始贴图片的位置
    :
    :return: 贴好的图片
    思路：1 将src_img_path图片的白色区域处理成透明颜色
          2 合并需要贴的图片和背景图片
    """

    src_img = cv2.imread(src_img_path)
    h, w = src_img.shape[:2]

    g, b, r = cv2.split(src_img)
    a = np.zeros_like(g) + 255
    # Todo 这块儿较耗时, 需要改进
    for wi in range(w):
        for hi in range(h):
            if np.all(src_img[wi, hi] >= np.array(alpha_thold)):
                g[wi, hi] = 0
                b[wi, hi] = 0
                r[wi, hi] = 0
                a[wi, hi] = 0

    m_h, m_w = g.shape[:2]

    dst_img = cv2.imread(dst_img_path)
    _x, _y = begin_pixel
    g1, b1, r1 = cv2.split(dst_img)

    m_list = []
    for chanel in zip((g, b, r), (g1, b1, r1)):
        chanel[1][_y: _y + m_h, _x: _x + m_w] = chanel[1][_y: _y + m_h, _x: _x + m_w] * ((255 - a) / 255) + np.array(
            chanel[0] * (a / 255), dtype=np.int8)
        m_list.append(chanel[1])

    return cv2.merge(m_list)

if __name__ == "__main__":
    img = combine_mask_to_image( "./images/mask_combine_src.jpg", "./images/mask_combine_dst.jpg")
    cv2.imwrite("mask_combine.jpg", img)