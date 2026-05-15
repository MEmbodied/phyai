:html_theme.sidebar_secondary.remove:

.. raw:: html

   <section class="phyai-hero">
     <div class="phyai-hero-bg" aria-hidden="true">
       <span class="phyai-phi" style="--x:8%;  --y:18%; --s:5.5rem; --r:-12deg; --d:0s;">&phi;</span>
       <span class="phyai-phi" style="--x:88%; --y:14%; --s:4rem;   --r:18deg;  --d:1.4s;">&phi;</span>
       <span class="phyai-phi" style="--x:70%; --y:78%; --s:7rem;   --r:-6deg;  --d:2.8s;">&phi;</span>
       <span class="phyai-phi" style="--x:18%; --y:74%; --s:4.5rem; --r:24deg;  --d:0.8s;">&phi;</span>
       <span class="phyai-phi" style="--x:48%; --y:10%; --s:3rem;   --r:-20deg; --d:2s;">&phi;</span>
       <span class="phyai-phi" style="--x:55%; --y:88%; --s:3.5rem; --r:8deg;   --d:1s;">&phi;</span>
       <span class="phyai-phi" style="--x:32%; --y:42%; --s:2.5rem; --r:14deg;  --d:1.6s;">&phi;</span>
       <span class="phyai-phi" style="--x:78%; --y:48%; --s:3rem;   --r:-10deg; --d:0.4s;">&phi;</span>
     </div>
     <div class="phyai-hero-inner">
       <p class="phyai-eyebrow">PhyAI &middot; v0.1.0</p>
       <h1 class="phyai-headline">Physical AI,<br/>served.</h1>
       <p class="phyai-tagline">A Python-first inference and serving stack for modern accelerators.</p>
       <div class="phyai-cta">
         <a class="phyai-btn phyai-btn-primary" href="quick_start/index.html">Get started</a>
         <a class="phyai-btn phyai-btn-secondary" href="arch/index.html">Learn the architecture</a>
       </div>
       <p class="phyai-meta">Apache 2.0 &nbsp;&middot;&nbsp; Python 3.12+ &nbsp;&middot;&nbsp; uv workspace</p>
     </div>
   </section>

   <section class="phyai-marquee" aria-hidden="true">
     <div class="phyai-marquee-track">
       <span class="phyai-marquee-phi phyai-grad-1">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-2">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-3">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-4">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-1">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-2">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-3">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-4">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-1">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-2">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-3">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-4">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-1">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-2">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-3">&phi;</span>
       <span class="phyai-marquee-phi phyai-grad-4">&phi;</span>
     </div>
   </section>

   <section class="phyai-section">
     <h2>Build, serve, scale.</h2>
     <p class="phyai-section-tag">A modular workspace &mdash; kernels, runtime, server.</p>
   </section>

.. grid:: 1 2 2 3
   :gutter: 4
   :margin: 0 0 5 0

   .. grid-item-card:: Quick Start
      :link: quick_start/index
      :link-type: doc

      Install and run your first PhyAI model.

   .. grid-item-card:: Architecture
      :link: arch/index
      :link-type: doc

      How the workspace packages fit together.

   .. grid-item-card:: Python API
      :link: api/index
      :link-type: doc

      Reference for the public ``phyai`` Python API.

   .. grid-item-card:: Blogs
      :link: blogs/index
      :link-type: doc

      Engineering notes and deep-dives.

   .. grid-item-card:: Contribute
      :link: contribute/index
      :link-type: doc

      How to file issues and get reviewed.

   .. grid-item-card:: GitHub
      :link: https://github.com/MEmbodied/phyai

      Source code and issue tracker.

.. toctree::
   :hidden:

   Docs <docs>
   Blogs <blogs/index>
   Contribution <contribute/index>

.. toctree::
   :hidden:
   :caption: PhyAI Python API

   autoapi/phyai/index

.. toctree::
   :hidden:
   :caption: C++ API

   CppAPI/library_root
