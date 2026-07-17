export const metadata = {
  title: "CropDog — Auto-Crop MVP",
  description: "Throwaway validation MVP for the auto-crop algorithm",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily:
            "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          background: "#f5f5f5",
          color: "#111",
        }}
      >
        {children}
      </body>
    </html>
  );
}
