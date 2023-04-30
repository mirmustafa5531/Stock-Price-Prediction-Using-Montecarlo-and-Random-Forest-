from django.db import models

# Create your models here.


class Stocksearch(models.Model):
    stockname = models.CharField(max_length=50)
    company_name = models.CharField(max_length=50)
    