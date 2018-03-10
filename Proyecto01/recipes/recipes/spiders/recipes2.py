import scrapy
import unicodedata
from scrapy.selector import Selector
from scrapy.http import HtmlResponse

class QuotesSpider(scrapy.Spider):
    name = "recipes2"

    def start_requests(self):
        urls = []
        with open('ligas-recetas.txt') as f:
            urls = f.readlines()

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        # get information for each link
        title = response.xpath('//span[@class="o-AssetTitle__a-HeadlineText"]/text()').extract_first()
        ingredients = response.xpath('//div[@class="o-Ingredients__m-Body"]/ul/li/text()').extract()
        directions = response.xpath('//div[@class="o-Method__m-Body"]/p/text()').extract()

        recipe_direction = ""
        for direction in directions:
            # search the real direction for the recipe
            if(len(direction) > len(recipe_direction)):
                recipe_direction = direction.strip()

        yield {
            "name": title,
            "ingredients": ingredients,
            "directions": recipe_direction
        }
