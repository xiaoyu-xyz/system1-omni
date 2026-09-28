use omni_jev::{BoxError, Config};
use tokio::net::TcpListener;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let config = Config::from_env()?;
    let app = omni_jev::app(&config)?;
    let listener = TcpListener::bind(config.bind).await?;
    eprintln!("omni-jev listening on {}", listener.local_addr()?);
    eprintln!("forwarding to {}", config.backend_url);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{SignalKind, signal};
        match signal(SignalKind::terminate()) {
            Ok(mut sigterm) => {
                sigterm.recv().await;
            }
            Err(_) => std::future::pending().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = terminate => {}
    }
}
