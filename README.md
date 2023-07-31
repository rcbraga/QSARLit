# QSARLit

![Python 3.8](https://img.shields.io/badge/python-3.8-blue) ![Python 3.9](https://img.shields.io/badge/python-3.9-blue) ![Python 3.10](https://img.shields.io/badge/python-3.10-blue) ![Docker](https://img.shields.io/badge/docker-supported-brightgreen)

## Testar e Implementar

Para testar e implementar o projeto, siga as etapas abaixo:

1. Clone o repositório Git para o seu computador local.
2. Certifique-se de que você tenha o Docker e o Docker Compose instalados.
3. No terminal, navegue até a pasta raiz do projeto.
4. Para iniciar o servidor em modo de depuração, execute os seguintes comandos:

   ```
   docker compose up --build debugger
   docker compose up debugger
   ```
5. Para iniciar o servidor em modo de produção, execute os seguintes comandos:

   ```
   docker compose up --build production
   docker compose up production
   ```

Os comandos acima irão construir e iniciar o contêiner Docker para a aplicação.

```console
docker exec -it qsartlit-app bash
```
