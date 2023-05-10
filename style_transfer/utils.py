import cv2
import numpy as np
def transfer_color(src, dest, mask=None, alpha=1.0, invert_mask=False, src_to_grayscale=False):
    """
    Transfer Color using YIQ colorspace. Useful in preserving colors in style transfer.
    This method assumes inputs of shape [Height, Width, Channel] in BGR Color Space
    """
    if np.max(src)<=1:
        print(' TRANSFER COLOR WARNING: received fp32 src image, scaling')
        src = (src.copy()*255).astype(np.uint8)
    if np.max(dest)<=1:
        print(' TRANSFER COLOR WARNING: received fp32 dest image, scaling')
        dest = (dest.copy()*255).astype(np.uint8)
    src, dest = src.clip(0,255), dest.clip(0,255)

    # print(np.min(src), np.max(src), np.min(dest), np.max(dest))
        
    # Resize src to dest's size
    H,W,_ = src.shape 
    dest = cv2.resize(dest, dsize=(W, H), interpolation=cv2.INTER_LINEAR)

    if src_to_grayscale:
       src = cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


    dest_gray = cv2.cvtColor(dest, cv2.COLOR_BGR2GRAY)              #1 Extract the Destination's luminance
    src_yiq = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)                #2 Convert the Source from BGR to YIQ/YCbCr
    src_yiq[...,0] = dest_gray* alpha + src_yiq[...,0]*(1-alpha)    #3 Combine Destination's luminance and Source's IQ/CbCr
    
    if mask is not None:
        mask = cv2.resize(mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)
        mask = np.clip(mask,0,1)
        if invert_mask:
            mask = 1-mask
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        out = (out*mask + dest*(1-mask)).astype(np.uint8)
    else:
        out = cv2.cvtColor(src_yiq, cv2.COLOR_YCrCb2BGR).clip(0,255)
    return out
