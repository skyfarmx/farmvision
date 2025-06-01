from django.db import models
from django.contrib.auth.models import User


class Users(models.Model):

    id = models.AutoField(auto_created=True, primary_key=True, serialize=False)
    kat_id = models.ForeignKey(User, on_delete=models.CASCADE)
    first_name = models.CharField(max_length=250, verbose_name='Adı')
    father_name = models.CharField(max_length=250, verbose_name='Ata adı')
    last_name = models.CharField(max_length=250, verbose_name='Soyadı')
    city = models.CharField(max_length=250, verbose_name='Ülke ve ya Şehir')
    picture = models.ImageField(upload_to='assets/images',blank=True, null=True,verbose_name='image')
    phone = models.CharField(max_length=250, verbose_name='phone')
    departments = models.CharField(max_length=250, verbose_name='departments')
    gender = models.CharField(max_length=250, verbose_name='cinsi')
    bio = models.CharField(max_length=2500, verbose_name='bio')
    birthday = models.DateField(verbose_name='Doğum günü')
    address = models.CharField(max_length=2500, verbose_name='Ünvan')   
    website = models.URLField(max_length=250,default=None, verbose_name='Veb sayt')   

    def __str__(self):
        return self.first_name





#/var/www/data/d220046e-e471-4a82-832a-4af2509a24f5/images





