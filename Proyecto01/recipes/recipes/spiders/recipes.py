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

        # We want to remove this crap
        prefix = "//www.cookingchanneltv.com/recipes/"
        prefix2 = "//www.cookingchanneltv.com/recipes/packages/"
        prefix3 = "//www.cookingchanneltv.com/recipes/photos/"
        prefix4 = "//www.cookingchanneltv.com/recipes/a-z"

        recipes = []
        
        # Now to discriminate the crappy links from the ones we actually want
        # This is an atrocius piece of code, but it works.
        for link in links:
            if(link.startswith(prefix)):
                recipes.insert(0, link)

        recipes2 = []
        for link in recipes:
            if not link.startswith(prefix2):
                recipes2.insert(0, link)

        recipes3 = []
        for link in recipes2:
            if not link.startswith(prefix3):
                recipes3.insert(0, link)
                
        recipes4 = []
        for link in recipes3:
            if not link.startswith(prefix4):
                recipes4.insert(0, link)

        # Then we write them to a file
        filename = 'ligas-recetas.txt' 
        with open(filename, 'w') as f:
            for link in recipes4:
                f.write("%s\n" % link)

        
                            


    
