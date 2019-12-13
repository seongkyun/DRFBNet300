def th_selector(over_area, altitude):
    if altitude <= 10:
        th_x = over_area // 2 # 40
        th_y = over_area // 4 # 20
        return th_x, th_y
    elif altitude <=20:
        th_x = over_area // 8
        th_y = over_area // 20
        return th_x, th_y
    else:
        th_x = over_area // 20
        th_y = over_area // 40
        return th_x, th_y
    

def is_overlap_area(gt, box):
    #order: [start x, start y, end x, end y]
    if(gt[0]<=int(box[0]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[2]) and int(box[2])<=gt[2])\
    or (gt[0]<=int(box[0]) and int(box[0])<=gt[2])\
    or (int(box[0])<=gt[0] and gt[2]<=int(box[2])):
        return True
    else:
        return False

def lable_selector(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    if box_a[5] > box_b[5]:
        return box_a[4], box_a[5]
    else:
        return box_b[4], box_b[5]

def bigger_box(box_a, box_b):
    #order: [start x, start y, end x, end y, lable, score]
    lable, score = lable_selector(box_a, box_b)
    bigger_box = [min(box_a[0], box_b[0]), min(box_a[1], box_b[1])
    , max(box_a[2], box_b[2]), max(box_a[3], box_b[3])
    , lable, score]
    return bigger_box

def is_same_obj(box, r_box, over_area, th_x, th_y):
    #order: [start x, start y, end x, end y]

    r_x_dist = (r_box[2] - r_box[0])
    l_x_dist = (box[2] - box[0])
    small_th = over_area // 3

    r_mx = (r_box[0] + r_box[2]) // 2
    sy_dist = abs(r_box[1] - box[1]) 
    ey_dist = abs(r_box[3] - box[3])
    l_mx = (box[0] + box[2]) // 2

    # is y distance close?
    if sy_dist<th_y and ey_dist<th_y:
        # is x center distance close?
        if abs(l_mx - r_mx) < th_x:
            return True
        else:
            # is object too small?
            if l_x_dist < small_th and r_x_dist < small_th:
                #return True
                return False
            box_size = (box[2] - box[0]) * (box[3] - box[1])
            r_box_size = (r_box[2] - r_box[0]) * (r_box[3] - r_box[1])
            th_size = over_area * over_area * 4
            th_th = int(over_area*0.2)

            #if (box_size >= th_size) and (r_box_size >= th_size):
            #    return True
            # 
            if (box_size >= th_size) and (r_box_size >= th_size)\
            and (abs(box[2] - over_area*9)<th_th)\
            and (abs(r_box[0] - over_area*7)<th_th)\
            and r_box[0] < box[2]:
                return True
            return False
    else:
        return False

def get_close_obj(boxes, r_box, over_area, th_x, th_y):
    #order: [start x, start y, end x, end y, lable, score]

    # make the same object map
    obj_map = []
    new_obj = 0
    for j in range(len(boxes)):
        obj_map.append(is_same_obj(boxes[j], r_box, over_area, th_x, th_y))

    # change the existing object
    for j in range(len(obj_map)):
        new_obj += int(obj_map[j])
        if obj_map[j]:
            boxes[j] = bigger_box(r_box, boxes[j])
            break

    # add the none existing obj
    if new_obj == 0:
        boxes.append(r_box)

    return None