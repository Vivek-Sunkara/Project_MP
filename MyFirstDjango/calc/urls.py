from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('solver/', views.linear_programming_solver, name='linear_programming_solver'),
    path('simplex/', views.simplex_view, name='simplex'),
]
