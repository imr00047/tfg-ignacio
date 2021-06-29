#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xml.etree.ElementTree as ET
import numpy as np

import os, math
import enoki
import mitsuba
import cv2

mitsuba.set_variant(mitsuba.variants()[1])

from datetime import datetime 
from mitsuba.core.xml import (load_dict,load_string)
from mitsuba.render import (ImageBlock)
from mitsuba.core import (Bitmap,Struct)


# In[ ]:


def render(s,**kwargs):
    """
    Method for render an store a scene.
        s:              Scene to read. It can be a dictionary or a loaded scene.
        ================    Optional    ================
        sensor:         Sensor to renderer. By default the first (0).
        out:            Path where will be stored. By default 'output'.
        extension:      Supported image file formats [PNG, OpenEXR, RGBE, PFM, PPM, JPEG, TGA, BMP]. By default png.
        format:         Especify the pixel format [Y, YA, RGB, RGBA, XYZ, XYZA, XYZAW, MultiChannel]. By default RGBA.
        t:              Type of data for each element [UInt8, UInt16, UInt32, Float16, Float32]. By default UInt8.
        srgb_gamma:     Determine whether the bitmap uses an sRGB gamma encoding. By default False.
        normalize:      Specify that we want normalize the output.
    """
    print("Using '{}' variant".format(mitsuba.variant()))
    options = {
        "sensor" : 0,
        "out" : 'output',
        "extension" : "png",           # Extension to use
        "format" : 'RGBA',
        "t" : "Float16",
        "srgb_gamma" : False,
        "normalize" : False
    }
    options.update(kwargs)

    scene = load_dict(s) if type(s) == dict else load_string(s)

    sensor = scene.sensors()[options['sensor']]
    scene.integrator().render(scene,sensor)
    

    film = sensor.film()
    if options['extension'] == "exr" and options['out']:
        return storeChannel(film, 5, 7, options["out"])
    elif options['out']:
        dataImages = film.bitmap(raw=True).split()

        img = film.bitmap(raw=True).convert(
            getattr(Bitmap.PixelFormat, options["format"]),
            getattr(Struct.Type, options["t"]),
            srgb_gamma=options["srgb_gamma"])
        
        img.write(options['out']+'.'+options["extension"])
        return img
    
    return film.bitmap(raw=True).convert(
            getattr(Bitmap.PixelFormat, options["format"]),
            getattr(Struct.Type, options["t"]),
            srgb_gamma=options["srgb_gamma"])


# In[ ]:


def getCameraPositions(**kwargs):
    """Method for get multiples camera positions.
        ================    Optional    ================
        height:         Height of the camera in world coordinates. By default 0.
        distance:       Distance from the center. By default 1.
        divisions:      Number of positions to return. By default 1.
        up:             Vector up of the 3D-wordl. By default in Y-axis.
        center:         Center of the camera
        begin:          Initial degree epsilon of the camera, for know where start. By default 0.
    """
    options = {
        "height" : 0,       # height
        "distance" : 1,     # Distance from center
        "divisions" : 1,    # Number of positions of the camera
        "up" : [0,1,0],     # Vector up of the 3D-world
        "center": [0,0,0],  # Center
        "begin" : 0         # Initial degree where get the camera positions
    }
    options.update(kwargs)

    result = []
    templateArr = np.array(options["up"], dtype=float)
    indices = np.squeeze(np.where(templateArr == 0))
    delta = float(360/options["divisions"])


    for i in range(options["divisions"]):
        auxArr = templateArr.copy() * options["height"]

        x = math.cos(math.radians(i*delta+options["begin"]))
        y = math.sin(math.radians(i*delta+options["begin"]))

        auxArr[indices[0]] = x * options["distance"]
        auxArr[indices[1]] = y * options["distance"]

        for i in range(3):
            auxArr[i] += options["center"][i]

        result.append(auxArr)

    return np.array(result)


# In[ ]:


def getXML(path, **kwargs):
    """Method for get an XML.
        path:           Path of the file to load.
        ================    Optional    ================
        tags:           Tags to remove
    """
    import xml.etree.ElementTree as ET
    options = {
        "tags": []      # Tags to remove
    }
    options.update(kwargs)

    file = path + ".xml"
    tree = ET.parse(file)
    data = tree.getroot()

    i = 0
    size = len(data)

    while(i<size):
        for tag in options["tags"]:
            if(data[i].tag == tag):
                data.remove(data[i])
                i -= 1
                size -= 1
                break
        i += 1

    return data


# In[ ]:


def normalize(a, axis=-1, order=2):
    """Method for normalize a NumPy array to a unit vector, given arbitrary axes and optimal performance.
        a:           Array to normalize.
        axis:        If axis is an integer, it specifies the axis of x along which to compute the vector norms. If axis is a 2-tuple, it specifies the axes that hold 2-D matrices, and the matrix norms of these matrices are computed. If axis is None then either a vector norm (when x is 1-D) or a matrix norm (when x is 2-D) is returned. The default is -1.
        order:       Order of the norm. inf means numpy’s inf object. The default is 2.
    """
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


# In[ ]:


def storeChannel(film, channelStart, channelEnd, filename):
    """Method for store a film in disc, given the channel where they are located
        film:         Film obtained from a Mitsuba's render
        channelStart: Specify the first channel to store
        channelEnd:   Specify the last channel to store
        filename:     Specify the path where to store the film
    """
    bitmaps = film.bitmap(raw=True).split()
    arr = np.array(bitmaps[channelStart][1])

    for i in range(channelStart+1, channelEnd+1):
        elem = np.squeeze(np.array(bitmaps[i][1]))
        N = np.shape(arr)
        listN = list(N)
        listN[-1] += 1
        auxArr = np.zeros(listN,dtype=arr.dtype)
        auxArr[:,:,:-1] = arr
        auxArr[:,:,i - channelStart] = elem
        arr = auxArr

    norm_arr = normalize(arr) * 0.5 + 0.5

    img_final = cv2.normalize(norm_arr, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(filename + ".jpg",img_final)
    
    return img_final


# In[ ]:


filename = "untitled"
path = r"C:\Users\nacho\OneDrive\Escritorio\mitsuba-scenes\Landscape"

# Folder to store the images
outputFolder = "result/dataset"

# Integrators to use
use = [False,False,True,True]
drawOut = True
integrators = ["volpath", "volpathmis", "depth", "aov"]
fileOutputs = ["sha", "alb", "depth", "normal"]
formats = ["YA", "RGBA", "Y", "RGB"]
types = ["UInt8","UInt8","Float64","Float16"]
normalizes = [False, False, True, False]
extension = "png"

# LookAt parameters of the camera
up = "0,1,0"
target = "0, 75, 2"

# Sampler to use
sampler = "multijitter"
sample_count = 32

# Resolution of the camera
width = 512
height = 512

# Positions of the camera
useOriginalCamera = False # Use original camera of the scene [FAIL IF NOT EXISTS]
filenameOC = "orig"
positions = getCameraPositions(divisions = 50, distance = 15, height = 10, center = [float(num) for num in target.split(",")])
#positions = getCameraPositions(divisions = 1, distance = 5, height = 10, center = [float(num) for num in target.split(",")])

# Direction of the sun
direction = "0,-1, -0.5"
intensity = 1


# In[ ]:


# Check for errors
if not path:
    raise Exception('A path must be determine')
path = path.replace("\\","/")

if not filename:
    raise Exception('A filename must be determine')    
if(len(integrators) != len(use)):
    raise Exception('Len integrators to use must be equal to Len use')
if(len(integrators) != len(fileOutputs)):
    raise Exception('Len integrators to use must be equal to Len fileOutputs')
if(len(integrators) != len(formats)):
    raise Exception('Len integrators to use must be equal to Len formats')
if(len(integrators) != len(types)):
    raise Exception('Len integrators to use must be equal to Len types')
if(len(integrators) != len(normalizes)):
    raise Exception('Len integrators to use must be equal to Len normalizes')

if useOriginalCamera:
    positions = [None]
init_datetime = datetime.now()

# Get the XML for future changes (position camera, type of integrator, the resolution, the sampler, etc.)
data = getXML(path + "/" + filename,tags=["default"])


# Convert relative paths to absolute paths
stringsElements = data.findall('.//string')
for stringElement in stringsElements:
    if(stringElement.get("name") == "filename" and stringElement.get("value")[1] != ':'):
        stringElement.set("value", path + "/" + stringElement.get("value"))


        
# Set the emitter
for emitter in data.findall("emitter"):
    data.remove(emitter)
emitterElem = ET.SubElement(data, 'emitter')
emitterElem.set("type","directional")
directionElem = ET.SubElement(emitterElem, 'vector')
directionElem.set("name","direction")
directionElem.set("value",direction)
radianteElem = ET.SubElement(emitterElem, 'spectrum')
radianteElem.set("name","irradiance")
radianteElem.set("value","{}".format(intensity))

output_path = path + "/"
if(outputFolder):
    output_path = output_path + outputFolder + "/"
    if not os.path.exists( output_path ):
        os.makedirs( output_path )
    
idPosition = 0

sensor = data.find("sensor")

# Set the sampler
sensor.find("sampler").clear()
sensor.find("sampler").set("type", sampler)
samplerCElem = ET.SubElement(sensor.find("sampler"), "integer")
samplerCElem.set("name", "sample_count")
samplerCElem.set("value", str(sample_count))

# Set the resolution
sensor.find("film").clear()
sensor.find("film").set("type", "hdrfilm")
widthElem = ET.SubElement(sensor.find("film"), "integer")
widthElem.set("name","width")
widthElem.set("value",str(width))
heightElem = ET.SubElement(sensor.find("film"), "integer")
heightElem.set("name","height")
heightElem.set("value",str(height))

for pos in positions:
    # Set the position of the camera
    if not useOriginalCamera:        
        sensor.find("transform").clear()
        sensor.find("transform").set("name", "to_world")
        lookatElem = ET.SubElement(sensor.find("transform"), 'lookat')
        lookatElem.set("origin", ', '.join(map(str, pos)))
        lookatElem.set("target", target)
        lookatElem.set("up", up)
    
    idIntegrator = 0
    for integrator in integrators:
        if use[idIntegrator] == False:
            idIntegrator += 1
            continue
        
        # Set the integrator
        data.find("integrator").clear()
        data.find("integrator").set("type",integrator)
        if(integrator == "aov"):
            aovsElem = ET.SubElement(data.find("integrator"),'string')
            aovsElem.set("name", "aovs")
            aovsElem.set("value", "nn:sh_normal")
        if(integrator == "path"):
            depthElem = ET.SubElement(data.find("integrator"),'integer')
            depthElem.set("name", "max_depth")
            depthElem.set("value", "2")
        
        
        # Render / Write data
        output_name = output_path + fileOutputs[idIntegrator] + "_" + (str(idPosition) if not useOriginalCamera else filenameOC)
        img = np.array(render(ET.tostring(data).decode("utf-8"), 
               out= "" if (normalizes[idIntegrator]) else output_name,
               extension="exr" if integrator == "aov" else extension, 
               format=formats[idIntegrator],
               t=types[idIntegrator]))
        
        # Normalize?
        if(normalizes[idIntegrator]):
            alpha = np.array(render(ET.tostring(data).decode("utf-8"),
                       out="",
                       format="RGBA",
                       t="Float16"))[:,:,3]
            
            img = cv2.normalize(np.float32(img), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            np.subtract(255,img, out=img)
            cv2.imwrite(output_name + "." + extension,np.array(img * alpha,dtype=np.uint8))
        
        idIntegrator += 1
    
    if(drawOut == False):
        idPosition += 1
        continue
    # Draw the out image
    alb = cv2.imread(output_path+"alb_" + (str(idPosition) if not useOriginalCamera else filenameOC) + "." + extension).astype(np.float32)
    sha = cv2.imread(output_path+"sha_" + (str(idPosition) if not useOriginalCamera else filenameOC) + "." + extension).astype(np.float32)
    
    shaFactor = sha * (1 / np.amax(sha))
    out = np.zeros(alb.shape)

    for i in range(len(alb)):
        for j in range (len(alb[i])):
            for k in range(len(alb[i][j])):
                out[i][j][k] = alb[i][j][k] * shaFactor[i][j][k];
    cv2.imwrite(output_path+"out_" + (str(idPosition) if not useOriginalCamera else filenameOC) + "." + extension, out)
    idPosition += 1


# In[ ]:


delta = datetime.now() - init_datetime
minutes = delta.total_seconds() / 60
print("Se ha tardado {} minutos en renderizar todo".format(minutes))

