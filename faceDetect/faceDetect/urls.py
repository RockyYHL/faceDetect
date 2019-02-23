"""faceDetect URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, pathdeha
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls import url
from myapp import views
from myapp.api.dehaze_api import dehaze
from django.conf import settings


urlpatterns = [
    # url(r'^$', views.index, name='index'),
    # url(r'^p/(?P<article_id>[0-9]+)/$', views.detail, name='detail'),
    url(r'^register/$', views.register, name='register'),
    url('^$', views.my_login, name='my_login'),
    url(r'^login/$', views.my_login, name='my_login'),
    url(r'^detect/$', views.detect, name='detect'),
    url(r'^dehaze/$', views.dehaze, name='dehaze'),
    url(r'^logout/$', views.my_logout, name='my_logout'),

    url(r'^test/$', views.test, name='test'),
    url(r'^api/apitest/$', dehaze, name='dehaze'),


]
