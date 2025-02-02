---
layout: page
title: How to Contribute to DSAI
permalink: /how-to-contribute/
description: Thank you for considering contributing to DataScience AI ❤️
nav: false
nav_order: 3
horizontal: false
---

<style>
    h1 {
      background: linear-gradient(45deg, #4CAF50, #2196F3, #FF9800, #FF5252);
      -webkit-background-clip: text;
      color: transparent;
      display: inline; /* Ensures the inline display for the gradient to work on text */
    }
  </style>


<style>
    h2 {
      background: linear-gradient(90deg, rgba(2,0,36,1) 0%, rgba(9,9,121,1) 37%, rgba(0,212,255,1) 100%);
      -webkit-background-clip: text;
      color: transparent;
      display: inline; /* Ensures the inline display for the gradient to work on text */
    }
  </style>

<!-- <h2>1. On github directly</h2> <br> -->


#### Add Papers 📑

To add papers to the collection, navigate to the [_bibliography/papers.bib](https://github.com/TotalEnergiesCode/ds-deep-learning-benchmark-datascience/blob/main/_bibliography/papers.bib) file and include the BibTeX citation for each paper following the provided example:

```latex
@misc{cherepanova2023performancedriven,
      abbr={NeurIPS},
      selected={true},
      title={A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning}, 
      author={Valeriia Cherepanova and Roman Levin and Gowthami Somepalli and Jonas Geiping and C. Bayan Bruss and Andrew Gordon Wilson and Tom Goldstein and Micah Goldblum},
      year={2023},
      eprint={2311.05877},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      code={https://github.com/vcherepanova/tabular-feature-selection},
      pdf={https://openreview.net/forum?id=5BqDSw8r5j},
      slide={https://neurips.cc/media/neurips-2023/Slides/72816.pdf},
      altmetric={248277},
      dimensions={true},
      abstract={Academic tabular benchmarks often contain small sets of curated features. In contrast, 
      data scientists typically collect as many features as possible into their datasets, and even engineer new 
      features from existing ones.....},
      preview={OneNet.png}, /* Add an image illustrating the proposed method */
}
```

Enhance the basic BibTeX entry with the following properties:

- `preview={OneNet.png}`: Add an image illustrating the proposed method.
- `selected={true}`: Indicate that the paper will appear on the home page of DSAI.
- `code={https://github.com/vcherepanova/tabular-feature-selection}`: Include the URL to the online code or place it in the `/assets/code/` folder.
- `pdf={https://openreview.net/forum?id=5BqDSw8r5j}`: Add the URL to the paper's online PDF or place it in the `/assets/pdf/` folder.
- `slide={https://neurips.cc/media/neurips-2023/Slides/72816.pdf}`: Add the URL to the online slide or place it in the `/assets/pdf/` folder.

Your contribution is now complete ✅, and it will appear on the web interface shortly after running GitHub actions.

#### Create Posters 📋

Creating posters involves a straightforward process. Begin by placing your posts in the [_posts/](https://github.com/TotalEnergiesCode/ds-deep-learning-benchmark-datascience/blob/main/_bibliography/_posts/) directory, following the format `YYYY-MM-DD-your_post_name.md`. This structure allows for easy organization and retrieval of content. For your convenience, you can use existing examples like `_posts/2023-01-03-SVM.md` as a foundation to structure your own posts.

It's important to note that the content should be formatted using Markdown, a lightweight markup language. Markdown provides a simple and readable way to format text that can be easily converted to HTML. This allows for flexibility in your writing style, as Markdown supports various formatting options.

Moreover, to enhance the classification of your posts, you can include specific information at the beginning of your Markdown file. By specifying tags and categories, you make it easier to classify and categorize your content. This not only streamlines the organization of posts but also improves the overall accessibility and searchability of your poster collection. Tags and categories serve as metadata, providing additional context about the content and facilitating a more efficient browsing experience for users.

```
---
layout: post
title: OneNet
date: 2023-01-02 00:12:00-0400
description: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling
tags: Forecasting, EnergyStorage 
categories: EnergyStorage
related_posts: true
featured: true
giscus_comments: true
related_posts: true
datatable: true
featured: true
---
```

