document.addEventListener('DOMContentLoaded', () => {
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        video.play().catch(error => {
            console.log("Autoplay was prevented:", error);
            // Optional: Show a "Play" button if autoplay is blocked
        });
    });
});
