from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.core.mail import send_mail
from django.contrib import messages
from .service import license_plate_extract
from .form import SignInForm

@csrf_exempt    
def index(request):
    if request.method == 'POST':
        return JsonResponse(license_plate_extract(request))
    return render(request, "index.html")


def contact(request):
    if request.method == 'POST':
        name = request.POST["name"]
        email = request.POST["email"]
        message = request.POST["message"]
        recipient_list = {'ebartile@gmail.com'}
        send_mail("Info from" + name, message, email, recipient_list, fail_silently=False)
        messages.add_message(request, messages.SUCCESS, "Thank you for contacting us.We will reply your message in less than 24 hrs")
    return redirect("index")

# def signin(request):
#     if request.user.is_authenticated():
#         return redirect('index')
#     elif request.method == 'POST':
#         form = SignInForm(request.POST)
#         if not form.is_valid():
#             return redirect('index')
#         else:
#             try:
#                 if not form.cleaned_data.get('remember'):
#                     request.session.set_expiry(0)

#                 messages.add_message(request, messages.INFO, "Logged in successfully")
#                 return reverse('index')
#             except:
#                 messages.add_message(request, messages.WARNING, "Failed to login")
#                 return reverse('login')
#     else:
#         return redirect('index')
