---
layout: about
title: Spotlight Energy
permalink: /
#subtitle: Awesome Conference 
# profile:
#   align: right
#   image: prof_pic.jpg
  # image_circular: false # crops the image to make it circular
  # more_info: >
  #   <p>555 your office number</p>
  #   <p>123 your address street</p>
  #   <p>Your City, State 12345</p>

#news: true  # includes a list of news items
 # includes a list of the newest posts
selected_papers: true # includes a list of papers marked as "selected={true}"
#social: true  # includes social icons at the bottom of the page
latest_posts: true 
---

We bring the latest developments in machine and deep learning for Energy⚡️ domain. Discover cutting-edge methods showcased at renowned conferences like <a href='https://icml.cc/virtual/2023/index.html'>ICML</a>, <a href='https://papers.nips.cc/'>NeurIPS</a>, <a href='https://iclr.cc/virtual/2023/papers.html?filter=titles'>ICLR</a>, <a href='https://pes-gm.org/'>IEEE Power & Energy Society</a>, and others. Delve deeper with our insightful blogs that provide detailed explanations and analyses of these groundbreaking papers, making the complex world of AI in Energy accessible and engaging. Stay informed, stay ahead.


📣 Feel free to suggest useful energy application frameworks by contacting the editors (khalid Oublal, Emmanuel LE BORGNE, David Benhaiem) or read <a href='./how-to-contribute/'>CONTRIBUTING.md 🚀</a> to follow the main contributing guidelines. Your input is appreciated.



<div class="post">

  <!-- {% assign blog_name_size = site.blog_name | size %}
  {% assign blog_description_size = site.blog_description | size %} -->

  <!-- {% if blog_name_size > 0 or blog_description_size > 0 %}
  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
  {% endif %} -->

  {% if site.display_tags or site.display_categories %}
  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in site.display_tags %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i> <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if site.display_categories.size > 0 and site.display_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in site.display_categories %}
        <li>
          <i class="fa-solid fa-tag fa-sm"></i> <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
  {% endif %}