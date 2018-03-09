import scrapy
import unicodedata
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

class QuotesSpider(scrapy.Spider):
    name = "recipes2"

    def start_requests(self):
        urls = ['https://www.cookingchanneltv.com/recipes/giada-de-laurentiis/chicken-scallopine-with-sage-and-fontina-cheese-1948089']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        ingredients = response.xpath('//div[@class="o-Ingredients__m-Body"]/ul/li/text()').extract()
        for ingredient in ingredients:
            print "nuevo ingrediente ================================================\n"
            print ingredient
        
