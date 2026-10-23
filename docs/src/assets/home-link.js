// The sidebar logo leads back to the Newtrinos landing page (the site name still leads to the docs home).
document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("a.docs-logo").forEach(function (a) {
        a.href = "https://newtrinos-org.github.io/";
        a.title = "Newtrinos home";
    });
});
