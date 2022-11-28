import cv2
import numpy as np
import os
from skimage.morphology import reconstruction
import matplotlib as mpl
mpl.use('Agg')  # この行を追記
import matplotlib.pyplot as plt

def r_g_correction(img, alpha_rb=1.0):
    """
    RとBの補正
    :param img:入力画像[0,255]
    :param alpha_rb = 補正度合い
    :param img_correction:補正画像[0,1]
    """
    img_norm = img / 255.
    img_correction = np.zeros_like(img_norm)
    img_r_mean = np.mean(img_norm[:, :, 2])
    img_g_mean = np.mean(img_norm[:, :, 1])
    img_b_mean = np.mean(img_norm[:, :, 0])

    # R_G correction
    img_correction[:, :, 2] = img_norm[:, :, 2] + alpha_rb * (img_g_mean - img_r_mean) * (1. - img_norm[:, :, 2]) * img_norm[:, :, 1]
    img_correction[:, :, 1] = img_norm[:, :, 1]
    img_correction[:, :, 0] = img_norm[:, :, 0]
    # bule attenuation
    # img_correction[:, :, 0] = img_norm[:, :, 0] + alpha_rb*(img_g_mean - img_b_mean)*(1.-img_norm[:, :, 0])*img_norm[:, :, 1]
    print(np.max(img_correction[:, :, 2]))
    # correction = np.clip(255 * img_correction, 0, 255)
    # cv2.imshow("correction",correction.astype(np.uint8))
    # cv2.imwrite(save_path + name + "_correction" + ".png", correction.astype(np.uint8))

    return img_correction


def AGW(img, img_correction, alpha=0.5, L=10):
    """
    AGWの適用
    param:img R,B補正画像[0,255]
    :param img_correction:補正画像[0,1]
    """
    img_norm = img / 255.
    # height, width = img_norm.shape[0], img_norm.shape[1]
    # img_correction_padding = np.pad(img_correction, [[r, r], [r, r],[r, r]], mode='symmetric')

    img_r_mean = np.mean(img_correction[:, :, 2])
    img_g_mean = np.mean(img_correction[:, :, 1])
    img_b_mean = np.mean(img_correction[:, :, 0])

    img_all_mean = np.zeros_like(img_norm)
    img_all_mean[:, :, 2] = img_r_mean
    img_all_mean[:, :, 1] = img_g_mean
    img_all_mean[:, :, 0] = img_b_mean

    # mean filter
    kernel = np.ones(((2 * L + 1), (2 * L + 1)), np.float32) / ((2 * L + 1) * (2 * L + 1))
    RL_r = cv2.filter2D(img_correction[:, :, 2], -1, kernel)
    GL_g = cv2.filter2D(img_correction[:, :, 1], -1, kernel)
    BL_b = cv2.filter2D(img_correction[:, :, 0], -1, kernel)

    # RL_r = RL_r[r : r + height, r : r + width]
    # GL_g = GL_g[r : r + height, r : r + width]
    # BL_b = BL_b[r : r + height, r : r + width]

    img_sita = np.zeros_like(img_norm)
    # alpha fusion
    img_sita[:, :, 2] = alpha * img_all_mean[:, :, 2] + (1 - alpha) * RL_r
    img_sita[:, :, 1] = alpha * img_all_mean[:, :, 1] + (1 - alpha) * GL_g
    img_sita[:, :, 0] = alpha * img_all_mean[:, :, 0] + (1 - alpha) * BL_b

    # sita = np.clip(255 * img_sita, 0, 255)
    # cv2.imshow("sita",sita.astype(np.uint8))

    # mどうするか問題
    m = np.mean(img_sita)
    # m = np.mean(img_sita[:, :, 1])
    # m = img_sita[:, :, 1]
    # print(m)

    img_tilde = np.zeros_like(img_norm)
    img_tilde[:, :, 2] = (img_correction[:, :, 2] / img_sita[:, :, 2]) * m
    img_tilde[:, :, 1] = (img_correction[:, :, 1] / img_sita[:, :, 1]) * m
    img_tilde[:, :, 0] = (img_correction[:, :, 0] / img_sita[:, :, 0]) * m

    img_grayworld = img_correction * (np.expand_dims(img_all_mean[:, :, 1], axis=-1) / img_all_mean)

    tilde = np.clip(255 * img_tilde, 0, 255)
    # cv2.imshow("tilde", tilde.astype(np.uint8))
    # img_grayworld_img = np.clip(255 * img_grayworld, 0, 255)
    # cv2.imshow("grayworld", img_grayworld_img.astype(np.uint8))
    # cv2.imwrite(save_path + name + "_tilde" + ".png", tilde.astype(np.uint8))
    # cv2.imwrite(save_path + name + "_gray" + ".png", img_grayworld_img.astype(np.uint8))

    return img_tilde


def DHECI(img, img_AGW, alpha_c=0.1):
    """
    DHECI法の適用
    param:img 入力画像
    param:img_AGW 色かぶり除去画像
    param:alpha_c アルファブレンドの重み
    """
    img_norm = img / 255.
    img_AGW = img_AGW * 255.
    # img_AGW = min_max(img_AGW)
    img_AGW_round = np.round(img_AGW)
    height, width = img.shape[0], img.shape[1]

    # 明度
    I = (img_AGW[:, :, 0] + img_AGW[:, :, 1] + img_AGW[:, :, 2]) / 3
    I_round = np.round(I)

    # パディング処理
    r = 5
    I_padding = np.pad(I, [[r, r], [r, r]], mode='symmetric')

    d_IH_padding = np.zeros_like(I_padding)
    d_IV_padding = np.zeros_like(I_padding)

    for i in range(r, height + r):
        for j in range(r, width + r):
            d_IH_padding[i, j] = (I_padding[i + 1, j + 1] + 2 * I_padding[i + 1, j] + I_padding[i + 1, j - 1]) - (
                        I_padding[i - 1, j + 1] + 2 * I_padding[i - 1, j] + I_padding[i - 1, j - 1])
            d_IV_padding[i, j] = (I_padding[i + 1, j + 1] + 2 * I_padding[i, j + 1] + I_padding[i - 1, j + 1]) - (
                        I_padding[i + 1, j - 1] + 2 * I_padding[i, j - 1] + I_padding[i - 1, j - 1])

    d_IH = d_IH_padding[r: r + height, r: r + width]
    d_IV = d_IV_padding[r: r + height, r: r + width]

    # dih = np.clip(255 * d_IH, 0, 255)
    # cv2.imshow("ih", dih.astype(np.uint8))
    # div = np.clip(255 * d_IV, 0, 255)
    # cv2.imshow("iv", div.astype(np.uint8))

    d_I = np.sqrt(d_IH ** 2 + d_IV ** 2)
    # di = np.clip(255 * d_I, 0, 255)
    # cv2.imshow("i", di.astype(np.uint8))
    d_I_round = np.round(d_I)

    h_Id = np.zeros_like(range(256))

    for z in range(256):
        # h_Id += d_I_round[list(zip(*np.where(I_round == 100)))]
        h = d_I_round[np.where(I_round == z)]
        h = h.astype(np.float32)
        h_Id[z] = np.sum(h)

    # 彩度計算
    a = np.minimum(np.minimum(img_norm[:, :, 0], img_norm[:, :, 1]), img_norm[:, :, 2])
    S = (1 - ((3 / np.maximum((img_norm[:, :, 2] + img_norm[:, :, 1] + img_norm[:, :, 0]), 0.1)) * a)) * 255

    S_padding = np.pad(S, [[r, r], [r, r]], mode='symmetric')

    S_IH_padding = np.zeros_like(S_padding)
    S_IV_padding = np.zeros_like(S_padding)

    for i in range(r, height + r):
        for j in range(r, width + r):
            S_IH_padding[i, j] = (S_padding[i + 1, j + 1] + 2 * S_padding[i + 1, j] + S_padding[i + 1, j - 1]) - (
                        S_padding[i - 1, j + 1] + 2 * S_padding[i - 1, j] + S_padding[i - 1, j - 1])
            S_IV_padding[i, j] = (S_padding[i + 1, j + 1] + 2 * S_padding[i, j + 1] + S_padding[i - 1, j + 1]) - (
                        S_padding[i + 1, j - 1] + 2 * S_padding[i, j - 1] + S_padding[i - 1, j - 1])

    S_IH = S_IH_padding[r: r + height, r: r + width]
    S_IV = S_IV_padding[r: r + height, r: r + width]

    S_I = np.sqrt(S_IH ** 2 + S_IV ** 2)
    S_I_round = np.round(S_I)

    h_Sd = np.zeros_like(range(256))

    for k in range(256):
        mm = S_I_round[np.where(I_round == k)]
        mm = mm.astype(np.float32)
        h_Sd[k] = np.sum(mm)

    h_colord = alpha_c * h_Sd + (1 - alpha_c) * h_Id

    S_r = np.zeros_like(range(256), dtype=np.float32)

    for r in range(256):
        S_r[r] = 255 * (np.sum(h_colord[0:r]) / np.sum(h_colord))

    out = np.zeros_like(img)

    for v in range(height):
        for w in range(width):
            I_r = img_AGW_round[v, w, 2]
            I_g = img_AGW_round[v, w, 1]
            I_b = img_AGW_round[v, w, 0]
            out[v, w, 2] = S_r[I_r.astype(np.uint8)]
            out[v, w, 1] = S_r[I_g.astype(np.uint8)]
            out[v, w, 0] = S_r[I_b.astype(np.uint8)]

    ax = range(256)
    plt.plot(ax, S_r)
    plt.savefig("hoge.png")

    return out

def main(img):
    """
    メイン
    :param img:
    :return:out
    """
    #wongらの手法
    img_correction = r_g_correction(img)
    img_AGW = AGW(img, img_correction)
    wong = DHECI(img, img_AGW)

    return wong


if __name__ == "__main__":
    # data_path = "../data/images/"
    data_path = "./demo/"#入力画像が保存されているフォルダのパス
    save_path = "./result/"#保存するときの保存先フォルダのパス
    folder = os.listdir(data_path)

    for img_name in folder:
        print(img_name)
        img = cv2.imread(data_path + img_name)
        name, ext = os.path.splitext(img_name)
        wong = main(img)
        # out = np.clip(255 * out, 0, 255)
        # cv2.imwrite(save_path + name + "wong" + ".png", wong.astype(np.uint8))
        # cv2.namedWindow("orig", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("orig", img)
        # cv2.namedWindow("out", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("out", wong.astype(np.uint8))
        cv2.waitKey()
        # cv2.destroyAllWindows()
