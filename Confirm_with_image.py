import numpy as np
import cv2
import sys

MIN_MATCH_COUNT = 10


# Lowe's ratio testでデータの精度の低いマッチングを間引く
def runRatioTest(matches, ratio):
    good = []
    for m, n in matches:
        # 第一候補点(m)と第二候補点(n)の距離差がm.distance/ratio倍より
        # 離れているものは妥当な特徴点として残す
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# マッチングされた特徴点を用いて射影変換する
def Transform(qry_h, qry_w, qry_kp, trn_kp, good):
    # 対応が取れた特徴点の座標を取り出す
    qry_pts = np.float32( [qry_kp[m.queryIdx].pt for m in good] ).reshape(-1,1,2)
    trn_pts = np.float32( [trn_kp[m.trainIdx].pt for m in good] ).reshape(-1,1,2)

    # 射影変換行列(3x3)を求める
    H, mask = cv2.findHomography(qry_pts, trn_pts, cv2.RANSAC, 5.0)
    # ndarrayをリストに変換
    matchesMask = mask.ravel().tolist()

    # 射影変換
    pts = np.float32( [[0,0],[0,qry_h-1],[qry_w-1,qry_h-1],[qry_w-1,0]] ).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)

    # 射影変換後の座標群とinlinersを返す
    return dst, matchesMask


if __name__ == '__main__':
    matchesMask = None
    dst = None
    h = None
    w = None

    # 本来はこの辺でコマンド引数のチェックが必要

    # クエリ画像
    query = cv2.imread(sys.argv[1])
    h, w = query.shape[:2]
    # 訓練（テスト）対象画像
    train = cv2.imread(sys.argv[2])

    # A-KAZE検出器の生成
    akaze = cv2.AKAZE_create()

    # クエリ画像の特徴量の検出とベクトル計算
    query_kp, query_des = akaze.detectAndCompute(query, None)
    # 訓練画像の特徴量の検出とベクトル計算
    train_kp, train_des = akaze.detectAndCompute(train, None)

    # 特徴量のマッチング
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(query_des, train_des, k=2)

    # ratio testでデータを間引く
    good = runRatioTest(matches, 0.7)

    # MIN_MATCH_COUNT以上のマッチング数があったら射影変換して該当箇所を矩形で囲む
    if len(good) > MIN_MATCH_COUNT:
        dst, matchesMask = Transform(h, w, query_kp, train_kp, good)

        # 訓練画像は後でも使うのでコピーを使用
        train2 = train.copy()
        # 射影変換で求めたマッチング領域を短形で囲む
        # inlinersが少ないと正しく矩形描画されない
        train2 = cv2.polylines(train2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print('十分なマッチングが得られませんでした - %d/%d' % (len(good), MIN_MATCH_COUNT))
        sys.exit()

    # マッチングした特徴点を描画
    draw_params = dict(singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img_result = cv2.drawMatches(query, query_kp, train, train_kp, good, None, **draw_params)

    # 画面に出力
    cv2.imshow('Matche - 1', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # マッチングした領域を切り出して射影変換して個別に保存
    # 短形描画の時と異なり、訓練画像の射影変換座標群を元にクエリ画像のサイズに変形するため
    # 引き渡す引数は訓練画像の射影変換座標群、クエリ画像の座標群の順とする必要がある
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    H = cv2.getPerspectiveTransform(dst, pts)
    # 求めた座標で訓練画像を変換
    img_warp = cv2.warpPerspective(train, H, (w,h))

    # 画面に出力
    cv2.imshow('Matche - 2', img_warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    '''
     切り出した画像でもう一度マッチングする
     ここでは、チェックしたいポイントに限定した画像を用意する
    '''
    # 変数の初期化
    matchesMask = None
    dst = None

    # チェック画像の読み込み
    img_checked = cv2.imread(sys.argv[3])
    # 精度を上げるために拡大
    img_checked = cv2.resize(img_checked, None, interpolation=cv2.INTER_LINEAR, fx=2, fy=2)
    h, w = img_checked.shape[:2]
    img_warp = cv2.resize(img_warp, None, interpolation=cv2.INTER_LINEAR, fx=2, fy=2)

    # 特徴量の検出と特徴量ベクトルの計算
    img_checked_kp, img_checked_des = akaze.detectAndCompute(img_checked, None)
    img_warp_kp, img_warp_des = akaze.detectAndCompute(img_warp, None)

    # 特徴量のマッチング
    matches = bf.knnMatch(img_checked_des, img_warp_des, k=2)

    # ratio testでデータを間引く
    good = runRatioTest(matches, 0.7)

    # MIN_MATCH_COUNT以上のマッチング数があったら射影変換して該当箇所を矩形で囲む
    if len(good) > MIN_MATCH_COUNT:
        dst, matchesMask = Transform(h, w, img_checked_kp, img_warp_kp, good)

        img_warp = cv2.polylines(img_warp, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print('十分なマッチングが得られませんでした - %d%d' % (len(good), MIN_MATCH_COUNT))
        sys.exit()

    # マッチングした特徴点を描画
    draw_params = dict(singlePointColor = None,
                       matchesMask = matchesMask,
                       flags = 2)

    img_result = cv2.drawMatches(img_checked, img_checked_kp, img_warp, img_warp_kp, good, None, **draw_params)

    img_result = cv2.resize(img_result, None, interpolation=cv2.INTER_LINEAR, fx=0.5, fy=0.5)
    print(img_result.shape[:2])

    # 画面に出力
    cv2.imshow('matche - 3', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
