import scrapy
from scrapy.selector import Selector
from scrapy.http import HtmlResponse


class QuotesSpider(scrapy.Spider):
    name = "recipes"
    body = ''
    

    
    def start_requests(self):
        urls = ['https://www.cookingchanneltv.com/recipes/articles/100-traditional-italian-recipes']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
            
    def parse(self, response):
        page = response.xpath('//li[contains(@href, "/recipes")]/@href ').extract()
        print "AQUI DEBE DE ESTAR LA MIERDA"
        print page
        print "AQUI TERMINA LA MIERDA"
        # filename = 'recipes-%s.html' % page
        # with open(filename, 'wb') as f:
        #     f.write(response.body)
        # self.log('Saved file %s' % filename)

