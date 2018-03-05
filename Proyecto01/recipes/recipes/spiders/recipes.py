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


    def link(self, links, prefix):
        recipes = []
        for link in links:
            if not link.startswith(prefix):
                recipes.insert(0, link)
        return recipes
        

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

        # We remove links with prefixes that we do not want.
        recipes = self.link(recipes, prefix2)
        recipes = self.link(recipes, prefix3)
        recipes = self.link(recipes, prefix4)

        # Then we write them to a file
        filename = 'ligas-recetas.txt' 
        with open(filename, 'w') as f:
            for link in recipes:
                f.write("%s\n" % link)

        
                            


    
