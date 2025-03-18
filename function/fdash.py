import os
from PIL import Image
import time
import base64
import cv2
import imutils
import shutil
import numpy as np
import json
from pathlib import Path

# 定義允許的影像副檔名
allowed_img_exts = (".png", ".bmp", ".jpg", ".jpeg", ".gif")

def info_ms(key, path):
    my_file = Path(path)
    if my_file.is_file():
        with open(my_file, "r") as f:
            content = json.load(f)
    else:
        content = {'type': 'dl-methods', 'datasets': [{'id': 'OWOtest9487', 'status': 'NG', 'describe': 'red'},
                                                       {'id': 'OAOtest9487', 'status': 'OK', 'describe': 'red'}]}
    p = ""
    for i in range(len(content["datasets"])):
        if content["datasets"][i]["id"] == key:
            p = "<br />Id : " + content["datasets"][i]["id"] + "<br /> Status : " + content["datasets"][i]["status"] + "<br /> Describe :" + content["datasets"][i]["describe"] + "<br />"
    return p

def info_simple_ms(key, path):
    my_file = Path(path)
    if my_file.is_file():
        with open(my_file, "r") as f:
            content = json.load(f)
    else:
        content = {'type': 'dl-methods', 'datasets': [{'id': 'OWOtest9487', 'status': 'NG', 'describe': 'red'},
                                                       {'id': 'OAOtest9487', 'status': 'OK', 'describe': 'red'}]}
    p = ""
    for i in range(len(content["datasets"])):
        if content["datasets"][i]["id"] == key:
            p = " " + content["datasets"][i]["describe"]
    return p

def read_txt_labels(txt_file):
    """
    Read labels from txt annotation file
    :param txt_file: txt annotation file path
    :return: tag list
    """
    with open(txt_file, "r") as f:
        labels = []
        for line in f.readlines():
            label_data = line.strip().split(" ")
            class_id = int(label_data[0])
            # Parsing bounding box coordinates
            coordinates = [float(x) for x in label_data[1:]]
            labels.append([class_id, coordinates])
    return labels

def draw_labels(image, labels):
    """
    Draw segmentation region outlines on the image
    :param image: image
    :param labels: list of labels
    """
    for label in labels:
        class_id, coordinates = label
        # 將座標轉換為整數並組成多邊形座標點
        points = [(int(x * image.shape[1]), int(y * image.shape[0])) for x, y in zip(coordinates[::2], coordinates[1::2])]
        # 使用多邊形繪製分割區域輪廓 (綠色線條)
        cv2.polylines(image, [np.array(points)], True, (0, 255, 0), 2)

def yolo2imagesbase64(pimg, ptxt):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    image = cv2.imread(pimg)
    height, width, _  = image.shape
    labels = read_txt_labels(ptxt)
    draw_labels(image, labels)
    window_size = (width//2, height//2)
    image = cv2.resize(image, window_size)
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    image_x = int((window_size[0] - image.shape[1]) / 2)
    image_y = int((window_size[1] - image.shape[0]) / 2)
    background[image_y:image_y + image.shape[0], image_x:image_x + image.shape[1]] = image
    return image

def compress_images(input_dir, output_dir, quality=85):
    """
    Compress all images in the input directory and save them to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(allowed_img_exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            img = Image.open(input_path)
            width, height = img.size
            new_size = (width//4, height//4)
            resized_image = img.resize(new_size)
            resized_image.save(output_path, quality=quality)
            print(f'{filename} compressed and saved to {output_dir}')

def yolo2images(pimg, ptxt, out):
    """
    Restore the YOLO semantic segmentation txt annotation file to the original image
    """
    image = cv2.imread(pimg)
    height, width, _  = image.shape
    labels = read_txt_labels(ptxt)
    draw_labels(image, labels)
    window_size = (width, height)
    image = cv2.resize(image, window_size)
    background = np.zeros((window_size[1], window_size[0], 3), np.uint8)
    image_x = int((window_size[0] - image.shape[1]) / 2)
    image_y = int((window_size[1] - image.shape[0]) / 2)
    background[image_y:image_y + image.shape[0], image_x:image_x + image.shape[1]] = image
    cv2.imwrite(out, image)


def report_function_d(ymal_path, original_image, predict_folder, train_folder, html_path, pout):
    # 將傳入的目錄參數轉換為 Path 物件，避免使用 '/' 運算子時發生錯誤
    train_folder = Path(train_folder)
    original_image = Path(original_image)
    predict_folder = Path(predict_folder)
    pout = Path(pout)

    # 使用 Path 物件進行路徑組合
    predict_folder_labels = predict_folder / "labels"
    up_ymal_path = Path(ymal_path).parent
    up_ymal_path_key = up_ymal_path / "mes.json"
    ptxt = up_ymal_path / "labels"
    pimg = up_ymal_path / "images"

    rawdir = pout / "raw"
    resdir = pout / "res"

    if not pout.exists():
        os.makedirs(rawdir)
        os.makedirs(resdir)

    '''
    讀取 txt 標註檔與原始圖片
    '''
    ytxt = []
    yimg = []

    up_ymal_path_img_t = pimg / "train"
    up_ymal_path_img_v = pimg / "val"
    for filename in os.listdir(up_ymal_path_img_t):
        if filename.lower().endswith(allowed_img_exts):
            yimg.append(up_ymal_path_img_t / filename)
    for filename in os.listdir(up_ymal_path_img_v):
        if filename.lower().endswith(allowed_img_exts):
            yimg.append(up_ymal_path_img_v / filename)

    up_ymal_path_txt_t = ptxt / "train"
    up_ymal_path_txt_v = ptxt / "val"
    for filename in os.listdir(up_ymal_path_txt_t):
        if filename.endswith(".txt"):
            ytxt.append(up_ymal_path_txt_t / filename)
    for filename in os.listdir(up_ymal_path_txt_v):
        if filename.endswith(".txt"):
            ytxt.append(up_ymal_path_txt_v / filename)

    temd = []
    for i in range(len(ytxt)):
        shutil.copyfile(ytxt[i], str(rawdir / Path(ytxt[i]).name))
        tname = Path(ytxt[i]).stem
        temd.append(tname)

    for i in range(len(yimg)):
        shutil.copyfile(yimg[i], str(rawdir / Path(yimg[i]).name))

    # 針對 rawdir 中的圖片，依據 label 的 stem 找出實際的副檔名
    for filename in temd:
        simg = None
        for ext in allowed_img_exts:
            candidate = Path(rawdir) / (filename + ext)
            if candidate.exists():
                simg = candidate
                break
        if simg is None:
            continue
        stxt = Path(rawdir) / (filename + ".txt")
        # 修正：將輸出路徑改為 resdir，而非 rawdir
        sout = Path(resdir) / (filename + simg.suffix)
        yolo2images(pimg=str(simg), ptxt=str(stxt), out=str(sout))
    # time
    st = time.strftime("%Y.%m.%d", time.localtime())
    lt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    cont_head ="<!DOCTYPE html><head><meta charset='utf-8'><meta http-equiv='X-UA-Compatible' content='IE=edge'><meta name='viewport' content='width=device-width, initial-scale=1'><meta name='description' content='Report'><meta name='author' content='koala'><title>Report</title><style>html{font-family: sans-serif; line-height: 1.15; -ms-text-size-adjust: 100%; -webkit-text-size-adjust: 100%; scroll-behavior: smooth;} body{margin: 0;} article, aside, footer, header, nav, section{display: block;} h1{font-size: 2em; margin: 0.67em 0;} figcaption, figure, main{display: block;} figure{margin: 1em 40px;} hr{box-sizing: content-box; height: 0; overflow: visible;} pre{font-family: monospace, monospace; font-size: 1em;} a{background-color: transparent; -webkit-text-decoration-skip: objects;} a:active, a:hover{outline-width: 0;} abbr[title]{border-bottom: none; text-decoration: underline; text-decoration: underline dotted;} b, strong{font-weight: inherit;} b, strong{font-weight: bolder;} code, kbd, samp{font-family: monospace, monospace; font-size: 1em;} dfn{font-style: italic;} mark{background-color: #ff0; color: #000;} small{font-size: 80%;} sub, sup{font-size: 75%; line-height: 0; position: relative; vertical-align: baseline;} sub{bottom: -0.25em;} sup{top: -0.5em;} audio, video{display: inline-block;} audio:not([controls]){display: none; height: 0;} img{border-style: none; border-radius: 3px;} svg:not(:root){overflow: hidden;} button, input, optgroup, select, textarea{font-family: sans-serif; font-size: 100%; line-height: 1.15; margin: 0;} button, input{overflow: visible;} button, select{text-transform: none;} button, html [type='button'], [type='reset'], [type='submit']{-webkit-appearance: button;} button::-moz-focus-inner, [type='button']::-moz-focus-inner, [type='reset']::-moz-focus-inner, [type='submit']::-moz-focus-inner{border-style: none; padding: 0;} button:-moz-focusring, [type='button']:-moz-focusring, [type='reset']:-moz-focusring, [type='submit']:-moz-focusring{outline: 1px dotted ButtonText;} fieldset{border: 1px solid #c0c0c0; margin: 0 2px; padding: 0.35em 0.625em 0.75em;} legend{box-sizing: border-box; color: inherit; display: table; max-width: 100%; padding: 0; white-space: normal;} progress{display: inline-block; vertical-align: baseline;} textarea{overflow: auto;} [type='checkbox'], [type='radio']{box-sizing: border-box; padding: 0;} [type='number']::-webkit-inner-spin-button, [type='number']::-webkit-outer-spin-button{height: auto;} [type='search']{-webkit-appearance: textfield; outline-offset: -2px;} [type='search']::-webkit-search-cancel-button, [type='search']::-webkit-search-decoration{-webkit-appearance: none;} ::-webkit-file-upload-button{-webkit-appearance: button; font: inherit;} details, menu{display: block;} summary{display: list-item;} canvas{display: inline-block;} template{display: none;} [hidden]{display: none;} html{font-size: 62.5%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif;} body{font-size: 1.8rem; line-height: 1.618; max-width: 90em; margin: auto; color: #4a4a4a; background-color: #f9f9f9; padding: 13px;} @media (max-width:684px){body{font-size: 1.53rem;} } @media (max-width:382px){body{font-size: 1.35rem;} } h1, h2, h3, h4, h5, h6{line-height: 1.1; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, 'Noto Sans', sans-serif; font-weight: 700; margin-top: 3rem; margin-bottom: 1.5rem; overflow-wrap: break-word; word-wrap: break-word; -ms-word-break: break-all; word-break: break-word; -ms-hyphens: auto; -moz-hyphens: auto; -webkit-hyphens: auto; hyphens: auto;} h1{font-size: 2.35em;} h2{font-size: 2.00em;} h3{font-size: 1.75em;} h4{font-size: 1.5em;} h5{font-size: 1.25em;} h6{font-size: 1em;} p{margin-top: 0px; margin-bottom: 2.5rem;} small, sub, sup{font-size: 75%;} hr{border-color: #2c8898;} a{text-decoration: none; color: #2c8898;} a:hover{color: #982c61; border-bottom: 2px solid #4a4a4a;} ul{padding-left: 1.4em; margin-top: 0px; margin-bottom: 2.5rem;} li{margin-bottom: 0.4em;} blockquote{font-style: italic; margin-left: 1.5em; padding-left: 1em; border-left: 3px solid #2c8898;} img{height: auto; width: 70%; margin-top: 0px; margin-bottom: 2.5rem;} pre{background-color: #f1f1f1; display: block; padding: 1em; overflow-x: auto; margin-top: 0px; margin-bottom: 2.5rem;} code{font-size: 0.9em; padding: 0 0.5em; background-color: #f1f1f1; white-space: pre-wrap;} pre>code{padding: 0; background-color: transparent; white-space: pre;} table{text-align: justify; width: 100%; border-collapse: collapse;} td, th{padding: 0.5em; border-bottom: 1px solid #f1f1f1;} input, textarea{border: 1px solid #4a4a4a;} input:focus, textarea:focus{border: 1px solid #2c8898;} textarea{width: 100%;} .button, button, input[type='submit'], input[type='reset'], input[type='button']{display: inline-block; padding: 5px 10px; text-align: center; text-decoration: none; white-space: nowrap; background-color: #2c8898; color: #f9f9f9; border-radius: 1px; border: 1px solid #2c8898; cursor: pointer; box-sizing: border-box;} .button[disabled], button[disabled], input[type='submit'][disabled], input[type='reset'][disabled], input[type='button'][disabled]{cursor: default; opacity: .5;} .button:focus, .button:hover, button:focus, button:hover, input[type='submit']:focus, input[type='submit']:hover, input[type='reset']:focus, input[type='reset']:hover, input[type='button']:focus, input[type='button']:hover{background-color: #982c61; border-color: #982c61; color: #f9f9f9; outline: 0;} textarea, select, input[type]{color: #4a4a4a; padding: 6px 10px; margin-bottom: 10px; background-color: #f1f1f1; border: 1px solid #f1f1f1; border-radius: 4px; box-shadow: none; box-sizing: border-box;} textarea:focus, select:focus, input[type]:focus{border: 1px solid #2c8898; outline: 0;} input[type='checkbox']:focus{outline: 1px dotted #2c8898;} label, legend, fieldset{display: block; margin-bottom: .5rem; font-weight: 600;} .out{position: relative; width: 100%; height: 100%; overflow: hidden; display: flex;} .mm{width: 33%; margin: 5px;} .mm img{width: 100%; border-radius: 3px;} .mmi{width: 12.5%; height: auto; margin: 5px;} .mmi img{width: 100%; border-radius: 3px;}</style><style> .listnonema{margin: 0px;} .listnonema label{margin: 0px;} .aright{font-size: 2rem;; position: absolute; right: 30px;}</style><style>.shape-ex1{width: 100%; display: block; transition: all .3s linear;}.shape-ex1 ul{padding-inline-start: 0px; margin: 0rem;}.shape-ex1 li::marker{font-size: 0;}.shape-ex1 input[type='checkbox'], .shape-ex1 input[type='radio']{position: absolute; opacity: 0; pointer-events: none; top: 0;}.shape-ex1-list-box{min-width: max-content; width: 45%;}.shape-ex1-list{position: relative;}.shape-ex1-list-name{font-size: 2.5rem; padding-right: 1.25rem; padding-left: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(209 213 219); display: block; min-width: max-content; position: relative; margin: 0rem;}.shape-ex1-list-name label{margin: 0rem;}.shape-ex1-list-name li{margin: 0rem;}.shape-ex1-list-name::after{content: ''; width: 0; height: 0; border-top: 8px solid #000; border-left: 4px solid transparent; border-right: 4px solid transparent; position: absolute; top: calc(50% - 4px); right: 8px;}.shape-ex1-list-sec-box{display: none;}.shape-ex1-list-sec{}.shape-ex1-list-sec-name{font-size: 2rem; padding-right: 1.25rem; padding-left: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(229 231 235); min-width: max-content; display: block; position: relative;}.shape-ex1-list-sec-name::after{content: ''; width: 0; height: 0; border-top: 8px solid #000; border-left: 4px solid transparent; border-right: 4px solid transparent; position: absolute; top: calc(50% - 4px); right: 6px;}.shape-ex1-list-thr-box{display: none;}.shape-ex1-list-thr-name{font-size: 1.5rem; padding-left: 0.625rem; padding-right: 0.625rem; padding-top: 0.25rem; padding-bottom: 0.25rem; background-color: rgb(243 244 246); min-width: max-content; display: block; position: relative;}.shape-ex1 input[type='checkbox']:checked+ul, .shape-ex1 input[type='radio']:checked+ul{display: block;}</style><style>.huge-btn{width: 45px;height: 45px;position: fixed;border-radius: 10px;}.topbtn{right: 20px;bottom: 20px;display: none;}.topbtnsize{font-size: 30px;}</style><style type='text/css'>.topscheader{z-index: 10000;background-color: #fff; padding: 15px 40px; height: 48px; position: fixed; top:0; left: 10%; width: 80%; box-sizing: border-box; transition: top 0.4s ease-in-out; box-shadow: 0 0 8px rgba(0, 0, 0, 0.2);}.hideUp .topscheader{top: -48px;}.line{width: 24px; height: 24px; background: #fff; cursor: pointer; display: block; padding: 16px; position: fixed; z-index: 2;}.line .menu,.line .menu::before,.line .menu::after{background: #222; content: ''; display: block; height: 1px; position: absolute; transition: background ease 0.15s, top ease 0.15s 0.15s, transform ease 0.15s; width: 20px;}.line .menu{left: 18px; top: 27px;}.line .menu::before{top: -6px;}.line .menu::after{top: 6px;}#menu:checked + .line .menu{background: transparent;}#menu:checked + .line .menu::before{transform: rotate(45deg);}#menu:checked + .line .menu::after{transform: rotate(-45deg);}#menu:checked + .line .menu::before,#menu:checked + .line .menu::after{top: 0; transition: top ease 0.15s, transform ease 0.15s 0.15s;}#menu:checked ~ .menu-list{width: 40%; overflow: auto;}.menu-list{width: 60px; height: 100vh; background: #fff; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1); padding-top: 50px; position: fixed; z-index: 1; transition: all 0.3s; overflow: hidden;}.menu-list ul{list-style: none; margin-left: 70px; margin-bottom: 200px; padding: 0;}.menu-list ul > li{display: block; width: 300px; margin: 5px;}input#menu{display: none;} .as {display:none;} #menu:checked ~ .menu-list .as { display: block;} .nav-box {position: fixed;left: 0;z-index: 100;}</style></head>"

    # con_main_header = "<body class='pagetop' onscroll='get()'><header class='topscheader'>Dashboard</header><header><h1 id='topgoal'>Report</h1><h5>Date. " + st + "</h5></header><main>"

    con_main_header_1 = "<body class='pagetop' onscroll='get()'><header class='topscheader'><b>Dashboard</b></header>"
    con_main_header_2 = "<header><h1 id='topgoal'>Report</h1><h5>Date. " + st + "</h5></header><main>"
    con_bar_h = "<div class='shape-ex1'><ul class='shape-ex1-list-box'><li class='shape-ex1-list listnonema'><label class='shape-ex1-list-name' for='ex1_1-checkbox'>YOLO <a href='#t1' class='aright'>HERE</a></label><input type='checkbox' name='ex1_1-input' id='ex1_1-checkbox' /><ul class='shape-ex1-list-sec-box'><li class='shape-ex1-list-sec listnonema'><label class='shape-ex1-list-sec-name' for='ex1_2_1-checkbox'>Train <a href='#t1w1' class='aright'>HERE</a></label><input type='radio' name='ex1_2-input' id='ex1_2_1-checkbox' /><ul class='shape-ex1-list-thr-box'>"

    # con_bar_m = "</ul></li><li class='shape-ex1-list-sec listnonema'><label class='shape-ex1-list-sec-name' for='ex1_2_2-checkbox'>Predict <a href='#t1w2' class='aright'>HERE</a></label><input type='radio' name='ex1_2-input' id='ex1_2_2-checkbox' /><ul class='shape-ex1-list-thr-box'>"

    con_bar_t = "</ul></li></ul></li></ul></div>"

    con_bar_c1 = ""
    con_bar_c2 = ""

    con_sides_h = "<nav class='nav-box'><input type='checkbox'id='menu'><label for='menu'class='line'><div class='menu'></div></label><div class='menu-list'><ul>"
    # <li> XXXX - <a href='#t1w1i" + filename + "' class='aright'>HERE</a></li>
    # <li> XXXX - <a href='#t1w2i" + filename + "' class='aright'>HERE</a></li>
    con_sides_m_1 = ""
    con_sides_m_2 = ""
    con_sides_t = "</ul></div></nav>"

    con_main_train_c = ""

    # <img src='data:image/png;base64, ' alt='... ' />
    for filename in os.listdir(train_folder):
        if filename.lower().endswith(allowed_img_exts):
            # tem_con_main_train_c = "<h4>" + filename + "</h4><p><img src='"+ train_folder + "/" + filename +"' alt='"+filename+"'></p>"
            # Opening an Image File
            image_path = train_folder / filename  # Replace with your image file path
            img = cv2.imread(str(image_path))
            h, w, _ = img.shape
            img_resize = imutils.resize(img, height=(h//2))
            h, w, _ = img_resize.shape
            print("H : ", h ,"; W : ", w)
            image = cv2.imencode('.png', img_resize)[1]
            base64_head = 'data:image/png;base64,'
            base64_main = str(base64.b64encode(image))[2:-1]
            image_code = base64_head + base64_main
            print("OpenCV resizes the image to base64 length ：%d" % len(image_code))
            # print(image_code)
            tem_con_main_train_c = "<h4 id='t1w1i" + filename + "'>" + filename + "</h4><p><img src='" + image_code + "' alt='" + filename + "' /></p><p><a href='#topgoal'>[TOP]</a></p>"
            tem_con_bar_c1 = "<li class='shape-ex1-list-thr-name listnonema'><span>" + filename + " <a href='#t1w1i" + filename + "' class='aright'>HERE</a></span></li>"
            tem_con_sides_m_1 = "<li><a href='#t1w1i" + filename + "' class='as'>" + filename + " </a></li>"
            con_sides_m_1 = con_sides_m_1 + tem_con_sides_m_1
            con_bar_c1 = con_bar_c1 + tem_con_bar_c1
            con_main_train_c = con_main_train_c + tem_con_main_train_c
            print(train_folder / filename)

# <li class='shape-ex1-list-thr-name listnonema'><span>Image 1 <a href='#t1w1i1' class='aright'>HERE</a></span></li>
# <h4 id='t1w1i2'>Image 2</h4><p><img src='./cow.jpg' alt='cow.jpg'></p>

    con_main_train_h = "<article><header><h1>Info.</h1></header><p>Machine learning image visual segmentation analysis. <br /> Dataset Path : " + str(ymal_path) + "<br /> Predict Folder Path : " + str(predict_folder) + "</p></article><article><header><h1 id='t1'>YOLO</h1></header><p>You Only Look Once (YOLO) is a state-of-the-art, real-time object detection algorithm introduced in 2015 by Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi in their famous research paper 'You Only Look Once: Unified, Real-Time Object Detection'. </p></article><article><header><h3 id='t1w1'>Train</h3></header><div><hr></div>"

    con_main_train_t = "</article>"
    # <header><h3 id='t1w2'>Predict</h3></header><div><hr></div>
    con_main_predict_h = "<article><header><h3 id='t1w2'>Predict</h3></header><div><hr></div>"

    files = []
    info_files = []
    files_check = []
    # input_folder = './yolo_runs_202X.../segment/predict'
    input_folder_labels = predict_folder_labels
    for filename in os.listdir(input_folder_labels):
        if filename.endswith((".txt")):
            info_files.append(input_folder_labels / filename)
            files.append(filename) 
            for con in files:
                files_check.append(Path(con).stem)
    print("INFO. Files - TXT : ", files)
    print("INFO. The File Of Number - TXT : ", len(files))
    print("INFO. TXT Path : ", info_files)

    images = []
    info_images = []
    images_check = []
    # 建立 predict_folder 中圖片的字典，鍵為 stem，值為完整檔名
    predict_images_dict = {}
    for filename in os.listdir(predict_folder):
        if filename.lower().endswith(allowed_img_exts):
            info_images.append(predict_folder / filename)
            images.append(filename)
            stem = Path(filename).stem
            predict_images_dict[stem] = filename
            images_check.append(stem)
    print("INFO. Images: ", images)
    print("INFO. The Images Of Number : ", len(images))
    print("INFO. Images Path : ", info_images)
    equal_lists = list(set(files_check).intersection(set(images_check)))
    print(equal_lists)
    con_main_predict_unit = ""

    # 建立 original_image 目錄中圖片的字典，鍵為 stem，值為完整檔名
    original_images_dict = {}
    for filename in os.listdir(original_image):
        if filename.lower().endswith(allowed_img_exts):
            original_images_dict[Path(filename).stem] = filename

    con_bar_m_1 = "</ul></li><li class='shape-ex1-list-sec listnonema'><label class='shape-ex1-list-sec-name' for='ex1_2_2-checkbox'>Predict (" + str(len(equal_lists)) + "/" + str(len(images)) + ")"
    con_bar_m_2 = " <a href='#t1w2' class='aright'>HERE</a></label><input type='radio' name='ex1_2-input' id='ex1_2_2-checkbox' /><ul class='shape-ex1-list-thr-box'>"

    con_bar_m = con_bar_m_1 + con_bar_m_2

    predict_images_path = []
    for stem in equal_lists:
        print(stem)
        if stem not in predict_images_dict:
            continue
        image_filename = predict_images_dict[stem]
        image_path = predict_folder / image_filename
        yplabels_path = predict_folder / "labels" / (stem + ".txt")
        if stem in original_images_dict:
            original_image_filename = original_images_dict[stem]
        else:
            original_image_filename = image_filename
        original_image_path = original_image / original_image_filename
        # Original image
        img = cv2.imread(str(original_image_path))
        h, w, _ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h, w, _ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png', img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        original_image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length ：%d" % len(original_image_code))
        # YOLO2images
        tar = resdir / (stem + ".png")
        img = cv2.imread(str(tar))
        h, w, _ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h, w, _ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png', img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        yoloraw_image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length ：%d" % len(yoloraw_image_code))
        # Predict
        img = cv2.imread(str(image_path))
        h, w, _ = img.shape
        img_resize = imutils.resize(img, height=(h//4))
        h, w, _ = img_resize.shape
        print("H : ", h ,"; W : ", w)
        image = cv2.imencode('.png', img_resize)[1]
        base64_head = 'data:image/png;base64,'
        base64_main = str(base64.b64encode(image))[2:-1]
        image_code = base64_head + base64_main
        print("OpenCV resizes the image to base64 length : %d" % len(image_code))
        # print(image_code)
        tem_con_bar_c2 = "<li class='shape-ex1-list-thr-name listnonema'><span>" + stem + " - " + info_simple_ms(stem, str(up_ymal_path_key)) + " <a href='#t1w2i" + stem + "' class='aright'>HERE</a></span></li>"

        tem_con_sides_m_2 = "<li><a href='#t1w2i" + stem + "' class='as'>" + stem + " - " + info_simple_ms(stem, str(up_ymal_path_key)) + " </a></li>"

        con_sides_m_2 = con_sides_m_2 + tem_con_sides_m_2

        con_bar_c2  = con_bar_c2 + tem_con_bar_c2
        tem_con_main_train_c = "<h4 id='t1w2i" + stem + "'>" + stem + "</h4><p><img src='" + image_code + "' alt='" + stem + "' /><a href='#topgoal'>[TOP]</a></p>"

        tem_con_main_predict_unit = "<h4 id='t1w2i" + stem + "'>" + stem + " - " + info_simple_ms(stem, str(up_ymal_path_key)) + " </h4><p><div class='out'><div class='mm'>Original image<br/><img src='" + original_image_code + "' alt='" + stem + " - Original image' /></div><div class='mm'>YOLO label image<br/><img src='" + yoloraw_image_code + "' alt='" + stem + " - YOLO Label Image' /></div><div class='mm'>YOLO training results image<br/><img src='" + image_code + "' alt='" + stem + " - YOLO Training Results Image' /></div></div>" + info_ms(stem, str(up_ymal_path_key)) + "<a href='#topgoal'>[TOP]</a><!-- IOU : --></p></article>"
        con_main_predict_unit = con_main_predict_unit + tem_con_main_predict_unit
        predict_images_path.append(image_path)
    # print(predict_images_path)

    # con_main_predict ="<h4>TEST.png</h4><p><div class='out'><div class='mm'>Original image<br/><img src='./TEST.png' alt='Original image'></div><div class='mm'>YOLO label image<br/><img src='./TEST.png' alt='YOLO label image'></div><div class='mm'>YOLO training results image<br/><img src='./TEST.png' alt='YOLO training results image'></div></div><!-- IOU : --></p></article></main>"

    con_sides_all = con_sides_h + con_sides_m_1 + con_sides_m_2 + con_sides_t

    con_bar_all = con_bar_h + con_bar_c1 + con_bar_m + con_bar_c2 + con_bar_t

    con_main_predict_t = "</main>"

    con_main_predict = con_main_predict_h + con_main_predict_unit + con_main_predict_t

    con_main = con_main_header_1 + con_sides_all + con_main_header_2 + con_bar_all + con_main_train_h + con_main_train_c + con_main_train_t + con_main_predict

    con_tail = "<footer>" + lt + "</footer></body></html><button class='huge-btn totop topbtn'><b class='topbtnsize'>^</b></button><script> function get(){var topverb1=document.documentElement.scrollTop;var topverb2=document.body.scrollHeight;if(topverb1>=topverb2/8){document.querySelector('.topbtn').style.display='inherit'}else{document.querySelector('.topbtn').style.display='none'}} document.querySelector('.totop').onclick=function(){document.querySelector('.pagetop').scrollIntoView(true)} </script><script type='text/javascript'> var bodyClass = document.body.classList, lastScrollY = 0; window.addEventListener('scroll', function(){ var st = this.scrollY; if( st < lastScrollY) { bodyClass.remove('hideUp');} else { bodyClass.add('hideUp');} lastScrollY = st;});</script></body></html>"

    con_all = cont_head + con_main + con_tail

    file = open(html_path, 'w')
    file.write(con_all)
    file.close()


# ymal_path = '/mnt/... /dataset.yaml'
# original_image = '/mnt/ ... /yolov8-datasets-predict-name'
# predict_folder = './yolo_runs_.../segment/predict'
# train_folder = './yolo_runs_.../segment/train'
# html_path = "./yolo_runs_.../segment/predict/index.html"
# pout = "./yolo_runs_.../yolo2images"
# report_function_d(ymal_path, original_image, predict_folder, train_folder, html_path, pout)

# # 統一基本路徑
# base_report_path = Path("D:/report-hand-yolo/default_data")
# base_yolo_path = Path("D:/YOLO_250101_TEM/yolo_runs_yolo11n-seg_dl_20250101")

# # 定義各個路徑
# yaml_path = base_report_path / "dataset_yolo" / "dataset.yaml"
# original_image_dir = base_report_path / "dataset_predict"
# predict_dir = base_yolo_path / "segment" / "predict"
# train_dir = base_yolo_path / "segment" / "train"
# html_file = base_yolo_path / "segment" / "index.html"
# pout_dir = base_yolo_path / "segment" / "yolo2images"

# # 呼叫報告函數
# report_function_d(yaml_path, original_image_dir, predict_dir, train_dir, html_file, pout_dir)
