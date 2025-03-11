import cv2
import numpy as np
import matplotlib.pyplot as plt
def hs(inp,ref):
    hin=cv2.calcHist([inp],[0],None,[256],[0,256])
    hre=cv2.calcHist([ref],[0],None,[256],[0,256])
    cdfin=hin.cumsum()/hin.sum()
    cdfre=hre.cumsum()/hre.sum()
    mapp=np.interp(cdfin,cdfre,np.arange(256))
    out=mapp[inp]
    return np.uint8(out)
inp = cv2.imread("images.jpeg")
ref = cv2.imread("download.jpeg")
if inp is None or ref is None:
    print("ERRor")
    exit()
out=hs(inp,ref)
hin=cv2.calcHist([inp],[0],None,[256],[0,256])
href=cv2.calcHist([ref],[0],None,[256],[0,256])
hout=cv2.calcHist([out],[0],None,[256],[0,256])
plt.subplot(2,3,1)
plt.imshow(inp,cmap='gray')
plt.title("input")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(ref,cmap='gray')
plt.title("reference")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(out,cmap='gray')
plt.title("output")
plt.axis("off")

plt.subplot(2,3,4)
plt.plot(hin,color='black')
plt.title("histinput")

plt.subplot(2,3,5)
plt.plot(href,color='black')
plt.title("histref")

plt.subplot(2,3,6)
plt.plot(hout,color='black')
plt.title("histoutput")

plt.tight_layout()
plt.show()
