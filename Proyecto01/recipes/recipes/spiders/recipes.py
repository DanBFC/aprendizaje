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
        # Note that its done with css, and not lxml
        links = response.css('a[href*=recipes]::attr(href)').extract()
        # this is just flow controll
        print "AQUI DEBE DE ESTAR LA MIERDA"
        # to actually see if it finds the recipes
        print links
        print "AQUI TERMINA LA MIERDA"
        # This is all to write the recipes to a file, once they have been found
        filename = 'ligas-recetas.txt'
        with open(filename, 'w') as f:
            for link in links:
                f.write("%s\n" % link)

