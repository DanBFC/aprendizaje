import scrapy
import unicodedata
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

class QuotesSpider(scrapy.Spider):
    name = "recipes"
    
    def start_requests(self):
        urls = ['https://www.cookingchanneltv.com/recipes/articles/100-traditional-italian-recipes']
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # Note that its done with css, and not lxml
        links = response.css('a[href*=recipes]::attr(href)').extract()

        # To actually see if it finds the recipes
        # This is just flow control
        print "AQUI DEBE DE ESTAR LA MIERDA"
        print links
        print "AQUI TERMINA LA MIERDA"

        # transform the whole list of unicode strings to string of utf-8
        for link in links:
            link.encode('utf-8')

        # We want to remove this crap
        prefix = "//www.cookingchanneltv.com/recipes/"
        recipes = []
        
        # Now to discriminate the crappy links from the ones we actually want.
        for link in links:
            if(link.startswith(prefix)):
                recipes.insert(0, link)

        # Then we write them to a file
        filename = 'ligas-recetas.txt' 
        with open(filename, 'w') as f:
            for link in recipes:
                f.write("%s\n" % link)

        
                            


    
