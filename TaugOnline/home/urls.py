from django.urls import path
from .views import index
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('', index, name='home'),
]

urlpatterns += staticfiles_urlpatterns()