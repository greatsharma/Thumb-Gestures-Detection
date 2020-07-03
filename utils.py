import cv2


def draw_text_with_backgroud(img, text, x, y, font_scale, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                            background=(0,0,0), foreground=(255,255,255), box_coords_1=(-5,5), box_coords_2=(5,-5)):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((x+box_coords_1[0], y+box_coords_1[1]), (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]))
    cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, fontScale=font_scale, color=foreground, thickness=thickness)
