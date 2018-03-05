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

    def is_valid(self, link):
        prefix0 = "//www.cookingchanneltv.com/recipes/" in link
        prefix1 = "//www.cookingchanneltv.com/recipes/packages/" not in link
        prefix2 = "//www.cookingchanneltv.com/recipes/photos/" not in link
        prefix3 = "//www.cookingchanneltv.com/recipes/a-z" not in link
        return prefix0 and prefix1 and prefix2 and prefix3
        #prefix4 = "//www.cookingchanneltv.com/recipes/packages/" not in somestring
        #prefix1 = "//www.cookingchanneltv.com/recipes/packages/" not in somestring


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
            if(self.is_valid(link)):
                recipes.insert(0, link)#.replace("//", "https://"))
            #if(link.startswith(prefix)):
            #    recipes.insert(0, link.replace("//", "https://"))

        # We remove links with prefixes that we do not want.
        #recipes = self.link(recipes, prefix2)
        #recipes = self.link(recipes, prefix3)
        #recipes = self.link(recipes, prefix4)

        # Then we write them to a file
        filename = 'ligas-recetas.txt'
        with open(filename, 'w') as f:
            for link in recipes:
                f.write("%s\n" % link)
