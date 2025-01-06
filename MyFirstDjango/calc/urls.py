from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('solve', views.solve_lp_graphical, name='solve_lp'),  # Updated to the correct view
    path('Graphical_method_of_solution_to_LP.html', views.graphical_method, name='graphical_method'),
]