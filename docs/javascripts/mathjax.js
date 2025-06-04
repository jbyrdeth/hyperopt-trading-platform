/* MathJax Configuration for HyperOpt Strategy Documentation */

window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true,
        packages: {
            '[+]': ['ams', 'physics', 'mhchem']
        }
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    },
    svg: {
        fontCache: 'global'
    },
    loader: {
        load: ['[tex]/ams', '[tex]/physics', '[tex]/mhchem']
    }
};

document$.subscribe(() => {
    MathJax.typesetPromise()
});

/* Custom styling for math elements */
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .MathJax {
            font-size: 1.1em !important;
        }
        
        .MathJax_Display {
            margin: 1.5em 0 !important;
            text-align: center !important;
        }
        
        .arithmatex {
            background: var(--md-code-bg-color);
            padding: 0.2em 0.4em;
            border-radius: 0.25em;
            font-family: var(--md-code-font-family);
        }
        
        .arithmatex .MathJax {
            background: transparent !important;
        }
        
        mjx-container[jax="SVG"] {
            direction: ltr;
        }
        
        mjx-container[jax="SVG"] > svg {
            overflow: visible;
            min-height: 1px;
            min-width: 1px;
        }
        
        mjx-container[jax="SVG"] > svg a {
            fill: blue;
            stroke: blue;
        }
    `;
    document.head.appendChild(style);
}); 