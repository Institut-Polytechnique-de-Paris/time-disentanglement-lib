---
layout: post
title: GAM config
description: Implementation of GAM and Hyper-prams tuning
tags: Forecasting
date: 2023-01-03 14:37:00-0400
#categories: CarbonEmissionModeling
featured: true
giscus_comments: true
related_posts: true
datatable: true
featured: true
---

{::nomarkdown}
{% assign jupyter_path = "assets/jupyter/GAM.ipynb" | relative_url %}
{% capture notebook_exists %}{% file_exists assets/jupyter/GAM.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
{% jupyter_notebook jupyter_path %}
{% else %}

<p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}
