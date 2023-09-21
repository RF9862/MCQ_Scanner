import cv2 as cv
import numpy as np
# from omr_post import post_processing
import os, json, sys, time
from mcq_post import post_processing
import datetime
from pathlib import Path
import getpass, shutil
import pytesseract
from pytesseract import Output

style_n = 2
ref_prob = 0.4
ths = 200

def border_set(img_, coor, tk, color):
    '''
    coor: [x0, x1, y0, y1] - this denotes border locations.
    tk: border thickness, color: border color.
    '''
    img = img_.copy()
    if coor[0] != None:
        img[:, coor[0]:coor[0]+tk] = color # left vertical
    if coor[1] != None:
        img[:, coor[1]-tk:coor[1]] = color # right vertical
    if coor[2] != None:                    
        img[coor[2]:coor[2]+tk,:] = color # up horizontal
    if coor[3] != None:
        img[coor[3]-tk:coor[3],:] = color # down horizontal          

    return img 
def subset(set, lim, loc):
    '''
    set: one or multi list or array, lim: size, loc:location(small, medi, large)
    This function reconstructs set according to size of lim in location of loc.
    '''
    cnt, len_set = 0, len(set)        
    v_coor_y1, index_ = [], []
    pop = []
    for i in range(len_set):
        if i < len_set-1:
            try:
                condition = set[i+1][0] - set[i][0]
            except:
                condition = set[i+1] - set[i]
            if condition < lim:
                cnt = cnt + 1
                pop.append(set[i])
            else:
                cnt = cnt + 1
                pop.append(set[i])
                pop = np.asarray(pop)
                try:
                    if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                    elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                    else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                except:
                    if loc == "small": v_coor_y1.append(min(pop))
                    elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                    else: v_coor_y1.append(max(pop))  
                index_.append(cnt)
                cnt = 0
                pop = []
        else:
            cnt += 1
            pop.append(set[i])
            pop = np.asarray(pop)
            try:
                if loc == "small": v_coor_y1.append([min(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
                elif loc == "medi": v_coor_y1.append([int(np.median(pop[:, 0])), min(pop[:, 1]), max(pop[:, 2])])
                else: v_coor_y1.append([max(pop[:, 0]), min(pop[:, 1]), max(pop[:, 2])])
            except:
                if loc == "small": v_coor_y1.append(min(pop))
                elif loc == "medi": v_coor_y1.append(int(np.median(pop)))
                else: v_coor_y1.append(max(pop))                    
            index_.append(cnt)

    return v_coor_y1, index_   

def line_remove(image):

    result = image.copy()
    try:
        gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image = cv.threshold(gray, ths, 255, cv.THRESH_BINARY)[1]
    except: pass
    
    thresh = 255 - image
    thresh = cv.dilate(thresh, np.ones((7,1)), iterations=1)

    # Remove horizontal lines
    horizontal_kernel = cv.getStructuringElement(cv.MORPH_RECT, (25,1))
    remove_horizontal = cv.morphologyEx(thresh, cv.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv.findContours(remove_horizontal, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        try: cv.drawContours(result, [c], -1, [255, 255, 255], 5)
        except: cv.drawContours(result, [c], -1, 255, 5)
            

    # Remove vertical lines
    vertical_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1,25))
    remove_vertical = cv.morphologyEx(thresh, cv.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv.findContours(remove_vertical, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        try: cv.drawContours(result, [c], -1, [255, 255, 255], 5)
        except: cv.drawContours(result, [c], -1, 255, 5)

    return result    

def loc_check(loc):
    lim = 15
    overcnt = 0
    k = 0
    for j in range(len(loc[0])):
        try:
            for i in range(j+1, len(loc[0])):
            
                if abs(loc[0][j]-loc[0][i]) < lim and abs(loc[1][j]-loc[1][i]) < lim:
                    overcnt = overcnt + 1
        except: pass
    return len(loc[0]) - overcnt
            

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    maxWidth = max(int(widthA), int(widthB))
    # maxWidth = max(int(np.linalg.norm(br-bl)), int(np.linalg.norm(tr-tl)))

    # compute the height of the new image, which will be the
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # maxHeight = max(int(np.linalg.norm(tr-br)), int(np.linalg.norm(tl-br)))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
        
def Match(template_path, gray, threshold, cnt):
    # if prop == "vertics": 
    _, bin = cv.threshold(gray, ths, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    # bin = gray.copy()
    template = cv.imread(str(template_path),0)
    _, template = cv.threshold(template, ths, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    tW, tH = template.shape[::-1]

    for k in np.linspace(0.5, 1.0, 4)[::-1]:
        temp_img = cv.resize(bin, None, fx = k, fy = k)
        imgcpy = cv.resize(gray, None, fx = k, fy = k)
        drawImg =temp_img.copy()
        res = cv.matchTemplate(temp_img ,template,cv.TM_CCOEFF_NORMED)
        loc = np.where( res >= threshold)
        
        # loc_cnt = loc_check(loc)
        loc_cnt = len(loc[0])
        if loc_cnt >= cnt: break

    if loc_cnt < cnt:
        for k in np.linspace(1.1, 1.4, 3):
            temp_img = cv.resize(bin, None, fx = k, fy = k)
            imgcpy = cv.resize(gray, None, fx = k, fy = k)
            
            res = cv.matchTemplate(temp_img ,template,cv.TM_CCOEFF_NORMED)
            loc = np.where( res >= threshold)
            loc_cnt = len(loc[0])
            if loc_cnt >= cnt: break  
    return loc, imgcpy, tW, tH

def getCoors(img, loc, tW, tH):
    coors = []
    drawImg =img.copy()
    for pt in zip(*loc[::-1]):
        # x_coors.append(pt[0])
        # y_coors.append(pt[1])
        coors.append([pt[0]+int(tW/2), pt[1]+int(tH/2)])
        cv.rectangle(drawImg, pt, (pt[0]+tW,pt[1]+tH), (0, 0, 255), 2)  
    # cv.imwrite("pink.jpg", drawImg)
    return coors   
def getRow_Col(coor, rlim):
    lim = 10
    coor.sort()

    srColuniq, c_cnt = subset(np.array(coor)[:,0], lim, 'medi')
    srRowuniq, r_cnt = subset(np.array(coor)[np.array(coor)[:, 1].argsort()][:, 1], lim, 'medi')
    rows = [srRowuniq[i] for i, v in enumerate(r_cnt) if v > rlim]
    cols = [srColuniq[i] for i, v in enumerate(c_cnt) if v > 4]

    return rows, cols   
def text_region(read_img, temp_img):
    '''
    read_img: main_image, temp_img: binary image
    This function removes points and lines noises, then gets exact text range.
    1. Set 4 node(node_size=6) of temp_img into 255
    2. Get only text regions in temp_img. (condition: h < 40 and w > tk and h > 8), save the image as temp
    3. Noise remove
    4. Get range including all texts from read_img

    '''
    img_h, img_w = temp_img.shape

    temp_img = border_set(temp_img, [0, img_w, 0, img_h], 1, 255)  
    cnt, _ = cv.findContours(temp_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    temp = np.zeros_like(temp_img)+255
    for c in cnt:
        x, y, w, h = cv.boundingRect(c)
        if h < min(temp_img.shape[0]*0.9, 35) and w > 4 and h > 10:# and w < 60:# and h >15:
            # cv.rectangle(xx, (x, y), (x + w, y + h), (0, 255, 0),1)   
            temp[y:y+h-1, x:x+w-1] = 0
    
    def xyRegion(temp):
        # Get range including all texts from read_img          
        kernel_hor = cv.getStructuringElement(cv.MORPH_RECT, (img_w, 1)) # vertical
        kernel_ver = cv.getStructuringElement(cv.MORPH_RECT, (1, img_h)) # vertical
        hor_temp = cv.erode(temp, kernel_hor, iterations=2)     
        ver_temp = cv.erode(temp, kernel_ver, iterations=2)
        img_vh = cv.addWeighted(ver_temp, 0.5, hor_temp, 0.5, 0.0)
        _, img_vh = cv.threshold(img_vh, 50, 255, cv.THRESH_BINARY)
        img_vh = border_set(img_vh, [0, img_w, 0, img_h], 2, 255)
        contours, _ = cv.findContours(img_vh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        x1, x2, y1, y2  = img_w, 0, img_h, 0
        for c in contours:
            x, y, w, h = cv.boundingRect(c) 
            if w < img_w and h < img_h:
                if x < x1: x1 = x
                if y < y1: y1 = y
                if x+w > x2: x2 = x+w
                if y+h > y2: y2 = y+h
        return x1,x2,y1,y2    

    x01,x02,y01,y02 = xyRegion(temp)            
    erod_size = 10
    temp = cv.erode(temp, np.ones((2,erod_size)), iterations=1) # 10 means letter space.
    temp = border_set(temp, [0, img_w, 0, img_h], 1, 255) 
    
    # noise remove     
    w_30 = False
    cnt, _ = cv.findContours(temp, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    ch_w = 15
    for c in cnt:
        x, y, w, h = cv.boundingRect(c)
        if w > ch_w:
            w_30 = True 
            break
    if w_30:
        for c in cnt:
            x, y, w, h = cv.boundingRect(c)
            if w < ch_w or h < 15: temp[y:y+h, x:x+w] = 255            

    # img_bin_ver = cv.erode(img_bin, np.ones((1,erod_size)), iterations=1)

    x1,x2,y1,y2 = xyRegion(temp)
    if x1 > 2: x1 = x1 + int(erod_size/2)
    if x2 < img_w -2: x2 = x2 - int(erod_size/2)
    x1, x2 = max(x1, x01), min(x2, x02)
    y1, y2 = max(y1, y01), min(y2, y02)

    img = read_img[y1:y2, x1:x2]
    pad = 10
    try: img = np.pad(img, ((pad, pad), (0,0)),mode='constant', constant_values=255) 
    except: img = np.pad(img, ((pad, pad), (pad,pad), (0,0)),mode='constant', constant_values=255) 

    return img             
def main(filename, template_path, tempPath, index):
    json_file = template_path/f"style.json"
    with open(json_file, "r") as f:
        template_info = json.load(f) 
    # template file processing ##
    idQue_xWratio = template_info["idQue_xWratio"]
    id_yHRatio = template_info["id_yHRatio"]
    ans_yHRatio = template_info["ans_yHRatio"]
    threshold = template_info["threshold"]

    src = cv.imread(filename, cv.IMREAD_COLOR)
    # img = line_remove(src)   ### binary image removed lines. 
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    ### getting every images for seat_id, question, answer marks ##
    id_img = gray[int(id_yHRatio*img_h):int(ans_yHRatio*img_h), 0:int(idQue_xWratio*img_w)]
    que_img = gray[int(id_yHRatio*img_h):int(ans_yHRatio*img_h), int(idQue_xWratio*img_w):]
    ans_img = gray[int(ans_yHRatio*img_h):]
    ### Finding all bubble in every region ###
    id_loc, id_img, idW, idH = Match(template_path/f"O_2.jpg", id_img, threshold, 50)
    q_loc, que_img, qW, qH = Match(template_path/f"O_2.jpg", que_img, threshold, 50)
    a_loc, ans_img, aW, aH = Match(template_path/f"O_2.jpg", ans_img, threshold, 50)
    ### seat number consideration ###
    try:
        id_coors = getCoors(id_img, id_loc, idW, idH)
        id_rows, id_cols = getRow_Col(id_coors, 1)
        st_num = SfindMark(id_rows, id_cols, id_img, idW, idH)
        st_num = ''.join(st_num)
        idCheckImg = id_img[id_rows[0]-120:id_rows[0]-40, id_cols[0]-30:id_cols[-1]+30]
        sheetNumImg = id_img[id_rows[0]-120:id_rows[0]+40, 0:id_cols[0]-50]
    except Exception as e:
        st_num = ""
        print(e)
    ### question number consideration ###
    try:
        q_coors = getCoors(que_img, q_loc, qW, qH)
        que_rows, que_cols = getRow_Col(q_coors, 1)
        que_num = SfindMark(que_rows, que_cols, que_img, idW, idH)
        que_num[0] = questionNum2letter(que_num[0])
        que_num = ''.join(que_num)
        queCheckImg = que_img[que_rows[0]-120:que_rows[0]-40, que_cols[0]-30:que_cols[-1]+30]
    except Exception as e:
        que_num = ""
        print(e)       
    ### Answer consideration ###
    try: 
        a_coors = getCoors(ans_img, a_loc, aW, aH)     
        ans_rows, ans_cols = getRow_Col(a_coors, 1)
        ans_num = AfindMark(ans_rows, ans_cols, 4, ans_img, aW, aH)
    except Exception as e:
        ans_num = []
        print(e)  

    ### get sheet number from sheetNumImg ###
    # _, temp_img = cv.threshold(sheetNumImg, 230, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # sheetNumImg = text_region(sheetNumImg, temp_img)
    sheetNum = pytesseract.image_to_string(sheetNumImg, config='--psm 11')
    sheetNum = ''.join([s for s in sheetNum.split() if s.isdigit()]).strip()

    nm = filename.split('/')[-1]
    sheet_img_path = f"{tempPath}/{index[0]}_{nm}"  # SHEET ID IMAGE
    id_img_path = f"{tempPath}/{index[1]}_{nm}"  # SEAT ID IMAGE
    que_img_path = f"{tempPath}/{index[2]}_{nm}"  # VERSION ID IMAGE
    
    writeTempImg(id_img_path, idCheckImg)
    writeTempImg(que_img_path, queCheckImg)
    writeTempImg(sheet_img_path, sheetNumImg)
    
    return [sheetNum, st_num, que_num,  ans_num]

def writeTempImg(path, img):
    flag = cv.imwrite(path, img)
    if not flag: cv.imwrite(path, np.ones([100,100,3],dtype=np.uint8)*255)


def questionNum2letter(num):
    if num == '1': return 'A'
    elif num == '2': return 'D'
    elif num == '3': return 'H'
    elif num == '4': return 'M'
    elif num == '5': return 'P'        

def SfindMark(rows, cols, img, sW, sH):
    _, img = cv.threshold(img, ths, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    # img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv.THRESH_BINARY,11,10)        
    out = []
    for col in cols:
        score = 75
        ind, check = filled_index_(img, rows, col, sH, sW, score, "st")

        while True:
            score = score + 10
            if not check: 
                ind, check = filled_index_(img, rows, col, sH, sW, score, "st")
            if check or score > 140: break
        if ind is None: ind = 'x'
        elif ind == '10': ind = '0'
        out.append(ind)
    return out
def AfindMark(rows, cols, sel_cnt, img, aW, aH):
    _, img = cv.threshold(img, ths, 255, cv.THRESH_BINARY | cv.THRESH_OTSU) 
    # img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #             cv.THRESH_BINARY,11,10)    
    cols_grp = [cols[i:i+sel_cnt] for i in range(0, len(cols),sel_cnt)]   
    out = []
    for cols in cols_grp:
        for row in rows:
            score = 85
            ind, check = filled_index_(img, cols, row, aH, aW, score, "ans")

            while True:
                score = score + 10
                if not check: 
                    ind, check = filled_index_(img, cols, row, aH, aW, score, "ans")
                if check or score > 150: break
            out.append(num2letter(ind))

    return out
def num2letter(ind):

    if ind == 1: return 'A'
    elif ind == 2: return 'B'
    elif ind == 3: return 'C'
    elif ind == 4: return 'D'
    elif ind == 5: return 'E'
    elif ind == 6: return 'F'
    elif ind is None: return '-'
        
def filled_index_(img, cols, row, aH, aW, score, prop):
    tk = 0
    if prop == "ans":
        for j, col in enumerate(cols):
            pic_img = img[row-int(aH/2)+tk:row+int(aH/2)-tk, col-int(aW/2)+tk:col+int(aW/2)-tk]
            if np.mean(pic_img) < score: 
                return j+1, True
    else:
        for j, col in enumerate(cols):
            pic_img = img[col-int(aH/2)+tk:col+int(aH/2)-tk, row-int(aW/2)+tk:row+int(aW/2)-tk]
            if np.mean(pic_img) < score: 
                return str(j+1), True        
    return None, False

def makedir(dir):
    try:
        os.mkdir(dir)
    except:
        pass     
def clear_contents(dir_path):
    '''
    Deletes the contents of the given filepath. Useful for testing runs.
    '''
    filelist = os.listdir(dir_path)
    if filelist:
        for f in filelist:
            if os.path.isdir(os.path.join(dir_path, f)):
                shutil.rmtree(os.path.join(dir_path, f))
            else:
                os.remove(os.path.join(dir_path, f))
    return None    
def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData
    
def data_check(input_dir, template_path):
    # check if style.json exist in template_path.
    if os.path.exists(os.path.join(template_path, 'style.json')):
        print("style.json file is not existed")
        return []
    
    ### getting user input directory ###
    '''
    Name of user input directory has to be 1,2,3,4, ...
    '''
    userDir = [stdir.parts[-1].split('_')[1] for stdir in styleDir if stdir.parts[-1].split('_')[0] == "style"]
    ### config directory check ###
    '''
    For every userDir, O_{}.jpg and template_{}.json files have to be existed in config directory 
    '''
    for ud in userDir:
        ud = ud
        O_filename = template_path/'_'.join(('O', ud+'.jpg'))
        template_filename = template_path/'_'.join(('style', ud+'.json'))
        if not O_filename.is_file():
            print("  File- " + '_'.join(('O', ud+'.jpg')) + " is not existed")
            return []

        if not template_filename.is_file():
            print("  File- " + '_'.join(('style', ud+'.json')) + " is not existed")
            return []
            
    return userDir         
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

#if __name__ == "__main__":
def mcq(main_self, data_dirs, out, cropPathBody, save_path, index):
    err_dir = out+"/failed"    
    # userPath = f"C:/Users/{getpass.getuser()}/.Mcq"
    # data_dir = Path(folder)
    template_path = Path(main_self.templatePath)
    temp_imgs = os.path.join(out, cropPathBody)
    makedir(temp_imgs)    
    makedir(err_dir)
    clear_contents(err_dir)
    if not os.path.exists(os.path.join(template_path, 'style.json')):
        print("style.json file is not existed")
    
    outputs = []
    cnt = 0
    fail_cnt = 0
    print("**********  Starting...  **********")
    start = datetime.datetime.now()
    # get total images for progress-bar 
    img_list = []
    for data_dir in data_dirs:
        img_list = img_list + [os.path.join(data_dir, f) for f in os.listdir(data_dir) if (f.split('.')[-1].lower() in ['png','jpg', 'tif'])]
    main_self.total1 = len(img_list)
    successImgs = []
    for im in img_list:
        im = im.replace("\\", "/")
        try:
            imBody = im.split('/')[-1]
            print(f"Parsing file_{cnt}: {im}")
            outputs.append(main(im, template_path, temp_imgs, index))
            cnt = cnt + 1
            main_self.cnt1=cnt
            main_self.single_done1.emit()
            successImgs.append(im)
        except Exception as e:
            fail_cnt += 1
            shutil.copyfile(im, os.path.join(err_dir, imBody))
            print(e)
            
    if len(outputs) > 0: 
        post_processing(outputs, save_path, successImgs, temp_imgs, index)
        print(f"Success: {len(img_list)-fail_cnt}, Failed: {fail_cnt}")
        duration = datetime.datetime.now() - start
        print(f"Time taken: {duration}") 
        return outputs, successImgs
    else: 
        print(f"Success: {len(img_list)-fail_cnt}, Failed: {fail_cnt}")
        duration = datetime.datetime.now() - start
        print(f"Time taken: {duration}")         
        return False
    