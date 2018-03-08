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

    def request_for_links(self, links):
        print "============================================================"
        print "Entra metodo"
        for link in links:
            yield scrapy.Request(url=link, callback=self.parse_link)

    def is_valid(self, link):
        # We want to remove this crap
        prefix0 = "//www.cookingchanneltv.com/recipes/" in link
        prefix1 = "//www.cookingchanneltv.com/recipes/packages/" not in link
        prefix2 = "//www.cookingchanneltv.com/recipes/photos/" not in link
        prefix3 = "//www.cookingchanneltv.com/recipes/a-z" not in link
        return prefix0 and prefix1 and prefix2 and prefix3


    def parse(self, response):
        # Note that its done with css, and not lxml
        links = response.css('a[href*=recipes]::attr(href)').extract()

        recipes = []

        # Now to discriminate the crappy links from the ones we actually want
        # This is an atrocius piece of code, but it works.
        for link in links:
            if(self.is_valid(link)):
                if("http" in link):
                    recipes.insert(0, link)
                else:
                    recipes.insert(0, link.replace("//", "https://"))

        print "============================================================"
        print "Antes del metodo"
        self.request_for_links([1,2,3,4])
        print "depues del metodo"

        # Then we write them to a file
        filename = 'ligas-recetas.txt'
        with open(filename, 'w') as f:
            for link in recipes:
                f.write("%s\n" % link)

    def parse_link(self, response):
        print "============================================================"
        print "============================================================"
        print "metodo"
        print response.status
        print "============================================================"
        print "============================================================"
