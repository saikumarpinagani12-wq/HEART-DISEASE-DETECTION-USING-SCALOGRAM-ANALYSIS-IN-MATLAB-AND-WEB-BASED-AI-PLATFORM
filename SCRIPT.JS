
const ecgFile = document.getElementById("ecgFile");
const fileName = document.getElementById("fileName");

if (ecgFile && fileName) {
    ecgFile.addEventListener("change", function () {
        if (this.files.length > 0) {
            fileName.textContent =
                "Selected: " + this.files[0].name;
        } else {
            fileName.textContent = "No file selected";
        }
    });
}

/* HeartCare AI homepage scroll animations */
document.addEventListener("DOMContentLoaded", function () {

    const sections = document.querySelectorAll(
        ".home-about, .workflow-section, .home-cta, " +
        ".feature-card, .workflow-grid > *, .technology-strip"
    );

    sections.forEach(function (element) {
        element.classList.add("reveal-on-scroll");
    });

    const revealObserver = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
            if (entry.isIntersecting) {
                entry.target.classList.add("visible");
                revealObserver.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.15
    });

    sections.forEach(function (element) {
        revealObserver.observe(element);
    });

});

/* HeartCare AI interactive hero lighting */
document.addEventListener("DOMContentLoaded", function () {

    const hero = document.querySelector(".home-hero");

    if (!hero) return;

    hero.addEventListener("pointermove", function (event) {
        const rect = hero.getBoundingClientRect();

        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        hero.style.setProperty("--mouse-x", x + "px");
        hero.style.setProperty("--mouse-y", y + "px");
    });

    hero.addEventListener("pointerleave", function () {
        hero.style.removeProperty("--mouse-x");
        hero.style.removeProperty("--mouse-y");
    });

});

document.addEventListener("DOMContentLoaded", function () {
    const hero = document.querySelector(".home-hero");

    if (!hero) return;

    hero.addEventListener("pointermove", function (event) {
        const rect = hero.getBoundingClientRect();

        hero.style.setProperty(
            "--mouse-x",
            (event.clientX - rect.left) + "px"
        );

        hero.style.setProperty(
            "--mouse-y",
            (event.clientY - rect.top) + "px"
        );
    });
});

/* ==========================================
   HEARTCARE AI - HOMEPAGE INTERACTIONS
========================================== */

document.addEventListener("DOMContentLoaded", function () {

    // Scroll reveal
    const revealItems = document.querySelectorAll(
        ".home-about, .feature-card, .workflow-step, " +
        ".home-cta, .technology-strip, .home-footer"
    );

    revealItems.forEach(function (item) {
        item.classList.add("hc-scroll-reveal");
    });

    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver(function (entries) {
            entries.forEach(function (entry) {
                if (entry.isIntersecting) {
                    entry.target.classList.add("hc-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });

        revealItems.forEach(function (item) {
            observer.observe(item);
        });
    } else {
        revealItems.forEach(function (item) {
            item.classList.add("hc-visible");
        });
    }

    // Mouse-following light
    const hero = document.querySelector(".home-hero");

    if (hero) {
        hero.addEventListener("pointermove", function (event) {
            const rect = hero.getBoundingClientRect();

            hero.style.setProperty(
                "--mouse-x",
                (event.clientX - rect.left) + "px"
            );

            hero.style.setProperty(
                "--mouse-y",
                (event.clientY - rect.top) + "px"
            );
        });

        hero.addEventListener("pointerleave", function () {
            hero.style.removeProperty("--mouse-x");
            hero.style.removeProperty("--mouse-y");
        });
    }

});

/* HeartCare AI - Interactive 3D feature cards */
document.addEventListener("DOMContentLoaded", function () {

    const cards = document.querySelectorAll(
        ".feature-card, .workflow-step"
    );

    const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    if (reduceMotion) return;

    cards.forEach(function (card) {

        card.addEventListener("pointermove", function (event) {

            const rect = card.getBoundingClientRect();

            const x = event.clientX - rect.left;
            const y = event.clientY - rect.top;

            const rotateY = ((x / rect.width) - 0.5) * 5;
            const rotateX = ((0.5 - y / rect.height)) * 5;

            card.style.transform =
                `perspective(800px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
        });

        card.addEventListener("pointerleave", function () {
            card.style.transform = "";
        });

    });

});

document.addEventListener("DOMContentLoaded", function () {

    /* Smooth reveal while scrolling */
    const revealItems = document.querySelectorAll(
        ".home-about, .feature-card, .workflow-step, " +
        ".home-cta, .technology-strip, .home-footer"
    );

    revealItems.forEach(item => {
        item.classList.add("hc-scroll-reveal");
    });

    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver(entries => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add("hc-visible");
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.15 });

        revealItems.forEach(item => observer.observe(item));
    } else {
        revealItems.forEach(item => item.classList.add("hc-visible"));
    }

    /* Subtle 3D card movement */
    const cards = document.querySelectorAll(
        ".feature-card, .workflow-step"
    );

    const reduceMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    if (!reduceMotion) {
        cards.forEach(card => {
            card.addEventListener("pointermove", event => {
                const rect = card.getBoundingClientRect();
                const x = (event.clientX - rect.left) / rect.width;
                const y = (event.clientY - rect.top) / rect.height;

                const rotateY = (x - 0.5) * 7;
                const rotateX = (0.5 - y) * 7;

                card.style.transform =
                    `perspective(900px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) translateY(-5px)`;
            });

            card.addEventListener("pointerleave", () => {
                card.style.transform = "";
            });
        });
    }

});
