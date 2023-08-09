// Ajoutez ce code à votre fichier main.js
document.addEventListener('DOMContentLoaded', () => {
  const customCursor = document.createElement('div');
  customCursor.id = 'custom-cursor';
  document.body.appendChild(customCursor);

  document.addEventListener('mousemove', (event) => {
    const cursor = document.getElementById('custom-cursor');
    cursor.style.left = `${event.clientX}px`;
    cursor.style.top = `${event.clientY}px`;
  });
});
