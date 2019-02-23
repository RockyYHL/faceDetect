# -*- coding: utf-8 -*-
from myapp.call_dehaze_api import Dehaze
import cv2
import datetime
import base64
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
import traceback

@api_view(['POST'])
@csrf_exempt
def dehaze(request):
    startTime = datetime.datetime.now()
    response_body = {}
    try:
        img_64 = request.data["img"]
        # print(type(img))
        img = base64.b64decode(img_64)
        with open('fog.jpg', 'wb') as f:
            f.write(img)
        haze_img = cv2.imread('fog.jpg')
        if request.data["r"]== "":
            r = 7
        else:
            r = int(request.data["r"])
        dh = Dehaze(r)
        dehaze_img = dh.deHaze(haze_img/255.0)*255
        psnr = dh.compute_psnr1(haze_img,dehaze_img)
        mssim = dh.compute_mssim(haze_img,dehaze_img)
        cv2.imwrite('defog.jpg', dehaze_img)
        with open('defog.jpg', 'rb') as f:
            dehaze_img_64 = base64.b64encode(f.read())
        response_body = {
            "result": 1,
            "msg": "执行成功",
            "data":{
                "dehaze_img_64":dehaze_img_64,
                "psnr": psnr,
                "mssim":mssim,
                "r":r,
            } ,
        }
        print("-----------接口调用成功--------------")
    except Exception as e:
        traceback.print_exc()
        response_body = {
            "result": 0,
            "msg": "执行失败",
            "data": ""
        }
    endTime = datetime.datetime.now()
    processTime = endTime - startTime
    response_body["data"]["processTime"] = processTime
    return Response(response_body)

