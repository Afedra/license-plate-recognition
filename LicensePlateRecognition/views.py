from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .service import license_plate_extract

@csrf_exempt    
def index(request):
    if request.method == 'POST':
        return JsonResponse(license_plate_extract(request))
    return render(request, "index.html")

